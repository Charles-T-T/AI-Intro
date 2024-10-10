# coding:utf-8
'''
    Lab 2
    井字棋(Tic tac toe)Python语言实现, 带有Alpha-Beta剪枝的Minimax算法.
'''
import random

# 棋盘位置表示（0-8）:
# 0  1  2
# 3  4  5
# 6  7  8

# 设定获胜的组合方式(横、竖、斜)
WINNING_TRIADS = ((0, 1, 2), (3, 4, 5), (6, 7, 8),
                  (0, 3, 6), (1, 4, 7), (2, 5, 8),
                  (0, 4, 8), (2, 4, 6))
# 设定棋盘按一行三个打印
PRINTING_TRIADS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
# 用一维列表表示棋盘:
SLOTS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
# -1表示X玩家 0表示空位 1表示O玩家.
X_token = -1
Open_token = 0
O_token = 1

MARKERS = ['_', 'O', 'X']
END_PHRASE = ('平局', '胜利', '失败')


def print_board(board):
    """ 打印当前棋盘 """
    for row in PRINTING_TRIADS:
        r = ' '
        for hole in row:
            r += MARKERS[board[hole]] + ' '
        print(r)


def legal_move_left(board):
    """ 判断棋盘上是否还有空位 """
    for slot in SLOTS:
        if board[slot] == Open_token:
            return True
    return False


def winner(board):
    """ 判断局面的胜者,返回值-1表示X获胜,1表示O获胜,0表示平局或者未结束 """
    for triad in WINNING_TRIADS:
        triad_sum = board[triad[0]] + board[triad[1]] + board[triad[2]]
        if triad_sum == 3 or triad_sum == -3:
            return board[triad[0]]  # 表示棋子的数值恰好也是-1:X,1:O
    return 0


def determine_move(board):
    """
        决定电脑(玩家O)的下一步棋(使用Alpha-beta 剪枝优化搜索效率)
        Args:
            board (list):井字棋盘

        Returns:
            next_move(int): 电脑(玩家O) 下一步棋的位置

    """
    def minimax(board, player, alpha, beta):
        """ 采用alpha-beta剪枝的minimax算法，返回当前棋盘的分数 """
        win = winner(board)
        if win != 0:
            return win  # 胜负已分
        if legal_move_left(board) == False:
            return 0  # 平局

        # 当前玩家为电脑（玩家O，max玩家）
        if player == O_token:
            max_val = -2  # 初始max值
            for slot in SLOTS:
                # 遍历所有可落子的位置
                if board[slot] == Open_token:
                    board[slot] = O_token  # 尝试落子
                    new_val = minimax(board, X_token, alpha, beta)  # 计算之后的分数
                    board[slot] = Open_token  # 还原棋盘
                    max_val = max(new_val, max_val)  # 更新max值
                    alpha = max(alpha, max_val)
                    if beta <= alpha:
                        break  # 剪枝
            return max_val

        # 当前玩家为人类（玩家X，min玩家）
        else:
            min_val = 2
            for slot in SLOTS:
                if board[slot] == Open_token:
                    board[slot] = X_token
                    new_val = minimax(board, O_token, alpha, beta)
                    board[slot] = Open_token
                    min_val = min(new_val, min_val)
                    beta = min(beta, min_val)
                    if alpha >= beta:
                        break
            return min_val

    # 初始化alpha和beta
    alpha = -2
    beta = 2
    max_val = -2

    # 计算不同落子位置对应的分数，取分数最大处落子
    for slot in SLOTS:
        if board[slot] == Open_token:
            board[slot] = O_token
            new_val = minimax(board, X_token, alpha, beta)
            if (new_val > max_val):
                max_val = new_val
                next_move = slot
            board[slot] = Open_token
    return next_move


HUMAN = 1
COMPUTER = 0


def main():
    """ 主函数,先决定谁是X(先手方),再开始下棋 """
    next_move = HUMAN
    opt = input("请选择先手方，输入X表示玩家先手，输入O表示电脑先手：")
    if opt == "X" or opt == "x":
        next_move = HUMAN
    elif opt == "O" or opt == "o":
        next_move = COMPUTER
    else:
        print("输入有误，默认玩家先手")

    # 初始化空棋盘
    board = [Open_token for i in range(9)]

    # 开始下棋
    while legal_move_left(board) and winner(board) == Open_token:
        print()
        print_board(board)
        if next_move == HUMAN and legal_move_left(board):
            try:
                print("\n")
                humanmv = int(input("请输入你要落子的位置(0-8)："))
                if board[humanmv] != Open_token:
                    continue
                board[humanmv] = X_token
                next_move = COMPUTER
            except:
                print("输入有误，请重试")
                continue
        elif next_move == COMPUTER and legal_move_left(board):  # 这里应该是elif
            mymv = determine_move(board)
            print("Computer最终决定下在", mymv)
            board[mymv] = O_token
            next_move = HUMAN

    # 输出结果
    print_board(board)
    print(["平局", "Computer赢了", "你赢了"][winner(board)])


if __name__ == '__main__':
    main()
