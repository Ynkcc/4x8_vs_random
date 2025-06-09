import random
from enum import Enum
import numpy as np
import copy

# 定义棋子类型的枚举
class PieceType(Enum):
    A = 0  # 兵/卒
    B = 1  # 炮/炮
    C = 2  # 马/傌
    D = 3  # 车/俥
    E = 4  # 象/相
    F = 5  # 士/仕
    G = 6  # 将/帅

class Piece:
    def __init__(self, piece_type, player):
        self.piece_type = piece_type
        self.player = player # -1 表示黑方, 1 表示红方
        self.revealed = False

    def __repr__(self):
        """提供棋子对象的字符串表示，方便调试。"""
        state = "R" if self.revealed else "H" # Revealed or Hidden
        player_char = "B" if self.player == -1 else "R" # Black or Red
        return f"{state}_{player_char}{self.piece_type.name}"

class GameEnvironment:
    # --- 类常量定义 ---
    BOARD_ROWS = 8
    BOARD_COLS = 4
    NUM_PIECE_TYPES = 7  # A 到 G
    TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS
    ACTION_SPACE_SIZE = TOTAL_POSITIONS * 5 # 每个位置: 上,下,左,右,翻

    MAX_CONSECUTIVE_MOVES = 40 # 最大连续未吃子/翻子移动步数，超过则平局
    WINNING_SCORE = 60         # 胜利所需分数

    PIECE_VALUES = {
        PieceType.A: 2, PieceType.B: 5, PieceType.C: 5,
        PieceType.D: 5, PieceType.E: 5, PieceType.F: 10, PieceType.G: 30
    }
    # 每种棋子的总数，用于归一化
    PIECE_MAX_COUNTS = {
        PieceType.A: 5, PieceType.B: 2, PieceType.C: 2,
        PieceType.D: 2, PieceType.E: 2, PieceType.F: 2, PieceType.G: 1
    }

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.board = np.empty((self.BOARD_ROWS, self.BOARD_COLS), dtype=object)
        self.dead_pieces = {-1: [], 1: []} # {-1: [被吃掉的黑棋], 1: [被吃掉的红棋]}
        self.current_player = 1 # 1 代表红方先手
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}
        self.reveal_counter = 0 # 新增：翻棋计数器
        
        # --- 用于优化 valid_actions 的数据结构 ---
        self.unrevealed_pieces_pos = set() # 存储所有未翻开棋子的位置 (r, c)
        self.player_pieces_pos = {-1: set(), 1: set()} # 存储各玩家已翻开棋子的位置

        self._initialize_board()
        
        self.state_size = (self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS * 2) + \
                          self.TOTAL_POSITIONS + (self.NUM_PIECE_TYPES * 2) + 2 + 1
                          
        self._state_vector = np.zeros(self.state_size, dtype=np.float32)

        self._my_pieces_plane_start_idx = 0
        self._opponent_pieces_plane_start_idx = self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS
        self._hidden_pieces_plane_start_idx = self._opponent_pieces_plane_start_idx + self.NUM_PIECE_TYPES * self.TOTAL_POSITIONS
        self._my_dead_count_start_idx = self._hidden_pieces_plane_start_idx + self.TOTAL_POSITIONS
        self._opponent_dead_count_start_idx = self._my_dead_count_start_idx + self.NUM_PIECE_TYPES
        self._scores_start_idx = self._opponent_dead_count_start_idx + self.NUM_PIECE_TYPES
        self._move_counter_idx = self._scores_start_idx + 2


    def _initialize_board(self):
        """
        初始化棋盘，随机放置双方的棋子，并填充位置追踪集合。
        """
        piece_counts = {
            PieceType.A: 5, PieceType.B: 2, PieceType.C: 2,
            PieceType.D: 2, PieceType.E: 2, PieceType.F: 2, PieceType.G: 1
        }
        pieces = []
        for piece_type, count in piece_counts.items():
            for _ in range(count):
                pieces.append(Piece(piece_type, -1)) # 黑方棋子
                pieces.append(Piece(piece_type, 1))  # 红方棋子

        random.shuffle(pieces)
        
        self.unrevealed_pieces_pos.clear()
        self.player_pieces_pos[-1].clear()
        self.player_pieces_pos[1].clear()

        idx = 0
        for row in range(self.BOARD_ROWS):
            for col in range(self.BOARD_COLS):
                self.board[row, col] = pieces[idx]
                self.unrevealed_pieces_pos.add((row, col))
                idx += 1
        
        self.dead_pieces = {-1: [], 1: []}
        self.current_player = 1 
        self.move_counter = 0
        self.scores = {-1: 0, 1: 0}
        self.reveal_counter = 0


    def reset(self):
        """
        重置游戏环境到初始状态。
        """
        self._initialize_board()
        return self.get_state()

    def get_state(self):
        """
        获取当前游戏状态，返回一个特征向量。
        """
        current_p = self.current_player
        opponent_p = -current_p

        self._state_vector.fill(0.0)

        for r, c in self.unrevealed_pieces_pos:
            pos_flat_idx = r * self.BOARD_COLS + c
            idx = self._hidden_pieces_plane_start_idx + pos_flat_idx
            self._state_vector[idx] = 1.0
        
        for r, c in self.player_pieces_pos[current_p]:
            piece = self.board[r, c]
            piece_type_idx = piece.piece_type.value
            pos_flat_idx = r * self.BOARD_COLS + c
            idx = self._my_pieces_plane_start_idx + piece_type_idx * self.TOTAL_POSITIONS + pos_flat_idx
            self._state_vector[idx] = 1.0

        for r, c in self.player_pieces_pos[opponent_p]:
            piece = self.board[r, c]
            piece_type_idx = piece.piece_type.value
            pos_flat_idx = r * self.BOARD_COLS + c
            idx = self._opponent_pieces_plane_start_idx + piece_type_idx * self.TOTAL_POSITIONS + pos_flat_idx
            self._state_vector[idx] = 1.0
            
        my_dead_counts = {pt: 0 for pt in PieceType}
        for p in self.dead_pieces[current_p]:
            my_dead_counts[p.piece_type] += 1
        for piece_type, count in my_dead_counts.items():
            max_count = self.PIECE_MAX_COUNTS[piece_type]
            idx = self._my_dead_count_start_idx + piece_type.value
            self._state_vector[idx] = count / max_count if max_count > 0 else 0

        opp_dead_counts = {pt: 0 for pt in PieceType}
        for p in self.dead_pieces[opponent_p]:
            opp_dead_counts[p.piece_type] += 1
        for piece_type, count in opp_dead_counts.items():
            max_count = self.PIECE_MAX_COUNTS[piece_type]
            idx = self._opponent_dead_count_start_idx + piece_type.value
            self._state_vector[idx] = count / max_count if max_count > 0 else 0
            
        score_norm = self.WINNING_SCORE if self.WINNING_SCORE > 0 else 1.0
        self._state_vector[self._scores_start_idx] = self.scores[current_p] / score_norm
        self._state_vector[self._scores_start_idx + 1] = self.scores[opponent_p] / score_norm

        move_norm = self.MAX_CONSECUTIVE_MOVES if self.MAX_CONSECUTIVE_MOVES > 0 else 1.0
        self._state_vector[self._move_counter_idx] = self.move_counter / move_norm
        
        return self._state_vector.copy()


    def step(self, action_index):
        """
        执行一个动作，更新游戏状态。
        """
        pos_idx = action_index // 5
        action_sub_idx = action_index % 5
        
        row = pos_idx // self.BOARD_COLS
        col = pos_idx % self.BOARD_COLS
        from_pos = (row, col)

        reward = 0.0
        info = {'winner': None}

        # 修改: 翻棋动作
        if action_sub_idx == 4:
            self.reveal(from_pos)
            self.move_counter = 0
            # 显著提高翻棋奖励，鼓励探索
            reward = 0.05
        else:
            d_row, d_col = 0, 0
            if action_sub_idx == 0: d_row = -1
            elif action_sub_idx == 1: d_row = 1
            elif action_sub_idx == 2: d_col = -1
            elif action_sub_idx == 3: d_col = 1
            
            attacker = self.board[row, col]
            
            # 炮的攻击逻辑
            if attacker.piece_type == PieceType.B:
                to_pos = self._get_cannon_target(from_pos, (d_row, d_col))
                if to_pos is None:
                    raise ValueError(f"炮从 {from_pos} 的移动/攻击路径无效")
                
                raw_reward = self.attack(from_pos, to_pos)
                reward = raw_reward / self.WINNING_SCORE if self.WINNING_SCORE > 0 else raw_reward
                self.move_counter = 0
            # 其他棋子的移动/攻击逻辑
            else:
                to_pos = (row + d_row, col + d_col)
                target = self.board[to_pos[0], to_pos[1]]
                # 移动到空格
                if target is None:
                    self.move(from_pos, to_pos)
                    self.move_counter += 1
                    # 修改: 移动到空格不给予奖励，避免无效移动
                    reward = 0.0
                # 攻击
                else:
                    raw_reward = self.attack(from_pos, to_pos)
                    reward = raw_reward / self.WINNING_SCORE if self.WINNING_SCORE > 0 else raw_reward
                    self.move_counter = 0
        
        done = False
        winner = None
        if self.scores[1] >= self.WINNING_SCORE:
            winner = 1
            done = True

        elif self.scores[-1] >= self.WINNING_SCORE:
            winner = -1
            done = True

        elif self.move_counter >= self.MAX_CONSECUTIVE_MOVES:
            done = True
            winner = 0


        self.current_player = -self.current_player
        if not done and np.sum(self.valid_actions()) == 0:
            done = True
            winner = -self.current_player


        info['winner'] = winner
        return self.get_state(), reward, done, info

    def reveal(self, position):
        """翻开指定位置的棋子。"""
        row, col = position
        piece_to_reveal = self.board[row, col]
        if piece_to_reveal is None:
            raise ValueError(f"尝试翻开空位置 ({row},{col})。")
        piece_to_reveal.revealed = True
        
        self.reveal_counter += 1
        
        self.unrevealed_pieces_pos.remove(position)
        self.player_pieces_pos[piece_to_reveal.player].add(position)


    def move(self, from_pos, to_pos):
        """移动棋子。"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        moving_piece = self.board[from_row, from_col]
        self.board[to_row, to_col] = moving_piece
        self.board[from_row, from_col] = None

        self.player_pieces_pos[moving_piece.player].remove(from_pos)
        self.player_pieces_pos[moving_piece.player].add(to_pos)


    def attack(self, from_pos, to_pos):
        """攻击并返回原始奖励值。"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        attacker = self.board[from_row, from_col]
        defender = self.board[to_row, to_col]
        
        self.dead_pieces[defender.player].append(defender)
        points = self.PIECE_VALUES[defender.piece_type]

        if attacker.player != defender.player:
            self.scores[attacker.player] += points
            reward = float(points)
        else: # 炮误伤己方暗棋
            self.scores[-attacker.player] += points
            reward = -float(points)

        # 修改: 移除对攻击暗棋的奖励清零逻辑
        if defender.revealed:
            self.player_pieces_pos[defender.player].remove(to_pos)
        else:
            self.unrevealed_pieces_pos.remove(to_pos)
            # 原有的 `reward = 0` 已被删除，使得攻击暗棋也能获得奖励/惩罚

        self.board[to_row, to_col] = attacker
        self.board[from_row, from_col] = None

        self.player_pieces_pos[attacker.player].remove(from_pos)
        self.player_pieces_pos[attacker.player].add(to_pos)
        
        return reward

    def can_attack(self, attacker, defender):
        """判断攻击方是否能吃掉防守方。"""
        if attacker.piece_type == PieceType.G and defender.piece_type == PieceType.A:
            return False
        if attacker.piece_type == PieceType.A and defender.piece_type == PieceType.G:
            return True
        if attacker.piece_type.value < defender.piece_type.value:
            return False
        return True
    
    def _get_cannon_target(self, from_pos, direction):
        """计算炮在指定方向上的攻击目标位置。"""
        dr, dc = direction
        r_check, c_check = from_pos
        mount_found = False
        while True:
            r_check += dr
            c_check += dc
            if not (0 <= r_check < self.BOARD_ROWS and 0 <= c_check < self.BOARD_COLS):
                return None
            
            target_on_path = self.board[r_check, c_check]
            if target_on_path:
                if not mount_found:
                    mount_found = True
                    continue
                else:
                    return (r_check, c_check)
        return None


    def valid_actions(self):
        """获取当前玩家所有合法的动作。"""
        valid_actions_arr = np.zeros(self.ACTION_SPACE_SIZE, dtype=int)
        current_p = self.current_player

        for r_from, c_from in self.unrevealed_pieces_pos:
            pos_idx = r_from * self.BOARD_COLS + c_from
            action_idx = pos_idx * 5 + 4
            valid_actions_arr[action_idx] = 1

        for r_from, c_from in self.player_pieces_pos[current_p]:
            piece = self.board[r_from, c_from]
            pos_idx = r_from * self.BOARD_COLS + c_from
            
            if piece.piece_type == PieceType.B:
                for move_sub_idx, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
                    target_pos = self._get_cannon_target((r_from, c_from), (dr, dc))
                    if target_pos:
                        target_piece = self.board[target_pos]
                        if not target_piece.revealed or target_piece.player != current_p:
                             action_idx = pos_idx * 5 + move_sub_idx
                             valid_actions_arr[action_idx] = 1
            else:
                for move_sub_idx, (dr, dc) in enumerate([(-1,0), (1,0), (0,-1), (0,1)]):
                    r_to, c_to = r_from + dr, c_from + dc
                    if not (0 <= r_to < self.BOARD_ROWS and 0 <= c_to < self.BOARD_COLS):
                        continue

                    action_idx = pos_idx * 5 + move_sub_idx
                    target_piece = self.board[r_to, c_to]

                    if target_piece is None:
                        valid_actions_arr[action_idx] = 1
                    elif target_piece.player != current_p and target_piece.revealed and self.can_attack(piece, target_piece):
                        valid_actions_arr[action_idx] = 1
        return valid_actions_arr