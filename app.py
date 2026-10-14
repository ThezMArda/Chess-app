import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# T490 CPU optimizasyonu: PyTorch thread sınırlandırması (Gecikmeyi önler)
torch.set_num_threads(4)
device = torch.device("cpu")
print(f"[*] Donanım: {device} (Optimize Edilmiş Arama Aktif)")

# --- MODEL MİMARİSİ ---
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.val_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.val_fc1 = nn.Linear(32 * 8 * 8, 64)
        self.val_fc2 = nn.Linear(64, 1)
        
        self.pol_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.pol_fc = nn.Linear(64 * 8 * 8, 64 * 64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        v = F.relu(self.val_conv(x)).view(x.size(0), -1)
        val = torch.tanh(self.val_fc2(F.relu(self.val_fc1(v))))
        
        p = F.relu(self.pol_conv(x)).view(x.size(0), -1)
        pol = self.pol_fc(p)
        return pol, val

MODEL_PATH = "rl_chess_model.pth"
model = ChessNet().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[✓] Model başarıyla yüklendi!")
else:
    raise FileNotFoundError(f"'{MODEL_PATH}' bulunamadı.")

def board_to_tensor(board: chess.Board):
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            offset = 0 if piece.color == chess.WHITE else 6
            channel = offset + piece_map[piece.piece_type]
            tensor[channel, square // 8, square % 8] = 1.0
            
    if board.turn == chess.BLACK:
        tensor = np.flip(tensor, axis=1).copy()
    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(device)

# --- MATERYAL DEĞERLERİ ---
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

def fast_material_eval(board: chess.Board):
    """Saf CPU seviyesinde mikro-saniyede çalışan materyal hesabı."""
    if board.is_checkmate():
        return -100000.0 if board.turn == chess.WHITE else 100000.0
    if board.is_game_over():
        return 0.0

    score = 0
    for square, piece in board.piece_map().items():
        val = PIECE_VALUES[piece.piece_type]
        score += val if piece.color == chess.WHITE else -val
    return score

def full_eval(board: chess.Board):
    """Minimax yapraklarında çalışan Model + Materyal hibriti."""
    mat = fast_material_eval(board)
    if abs(mat) > 50000:
        return mat

    state_tensor = board_to_tensor(board)
    with torch.no_grad():
        _, net_val = model(state_tensor)
        pos = net_val.item() * 120.0
        if board.turn == chess.BLACK:
            pos = -pos
    return mat + pos

def fast_quiescence(board: chess.Board, alpha: float, beta: float, is_maximizing: bool, max_depth: int = 2):
    """NN çağırmayan, sadece saf materyal alımlarına bakan ultra-hızlı sakinlik araması."""
    stand_pat = fast_material_eval(board)

    if is_maximizing:
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
    else:
        if stand_pat <= alpha:
            return alpha
        if beta > stand_pat:
            beta = stand_pat

    if max_depth == 0 or board.is_game_over():
        return stand_pat

    captures = [m for m in board.legal_moves if board.is_capture(m)]
    if not captures:
        return stand_pat

    # MVV-LVA sıralaması
    captures.sort(
        key=lambda m: PIECE_VALUES.get(board.piece_at(m.to_square).piece_type, 0) if board.piece_at(m.to_square) else 0,
        reverse=True
    )

    if is_maximizing:
        for move in captures:
            board.push(move)
            score = fast_quiescence(board, alpha, beta, False, max_depth - 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha
    else:
        for move in captures:
            board.push(move)
            score = fast_quiescence(board, alpha, beta, True, max_depth - 1)
            board.pop()
            if score <= alpha:
                return alpha
            if score < beta:
                beta = score
        return beta

def minimax(board: chess.Board, depth: int, alpha: float, beta: float, is_maximizing: bool):
    if board.is_game_over():
        return fast_material_eval(board), None

    if depth == 0:
        # Yaprakta sakinlik aramasıyla taktik körlüğü sıfırla
        return fast_quiescence(board, alpha, beta, is_maximizing, max_depth=2), None

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return fast_material_eval(board), None

    # Taş alma hamlelerini öne koy (Beta budaması anında çalışsın)
    legal_moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    best_move = None

    if is_maximizing:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

def get_best_move(board: chess.Board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, 0.0

    # 1. Kök pozisyonda sinir ağından hamle olasılıklarını al
    state_tensor = board_to_tensor(board)
    with torch.no_grad():
        policy, _ = model(state_tensor)
        policy = policy.squeeze(0).cpu().numpy()

    # 2. Hamleleri modelin önceliğine + taş alma durumuna göre sırala
    def move_priority(m):
        base_score = policy[m.from_square * 64 + m.to_square]
        if board.is_capture(m):
            base_score += 10.0  # Alımlara öncelik ver
        return base_score

    legal_moves.sort(key=move_priority, reverse=True)

    # Budama: En umut verici ilk 8 hamleyi 2 derinlikle tara (Hız patlaması!)
    candidate_moves = legal_moves[:8]

    best_move = candidate_moves[0]
    best_eval = float('inf') if board.turn == chess.BLACK else -float('inf')

    # Minimax kök döngüsü
    for move in candidate_moves:
        board.push(move)
        eval_score, _ = minimax(board, depth=1, alpha=-float('inf'), beta=float('inf'), is_maximizing=(board.turn == chess.WHITE))
        board.pop()

        if board.turn == chess.BLACK:
            if eval_score < best_eval:
                best_eval = eval_score
                best_move = move
        else:
            if eval_score > best_eval:
                best_eval = eval_score
                best_move = move

    return best_move, round(best_eval / 100.0, 2)

# --- FLASK ENDPOINTS ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/make_move", methods=["POST"])
def make_move():
    data = request.get_json()
    fen = data.get("fen")
    
    board = chess.Board(fen)
    if board.is_game_over():
        return jsonify({"game_over": True, "result": board.result()})

    ai_move, eval_score = get_best_move(board)
    if ai_move:
        board.push(ai_move)
        
    return jsonify({
        "fen": board.fen(),
        "ai_move": ai_move.uci() if ai_move else None,
        "evaluation": eval_score,
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)