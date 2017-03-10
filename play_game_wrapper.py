import time

from isolation import Board

from sample_players import GreedyPlayer

from game_agent import CustomPlayer

player_1 = CustomPlayer()

player_2 = GreedyPlayer()
#player_2 = RandomPlayer()

print(player_1,player_2)

test_game = Board(player_1, player_2)
start = time.time()
winner, moves, reason = test_game.play()
end = time.time()
#print (winner)
if reason == "timeout":
    print("Forfeit due to timeout.")
for move in moves:
    print(move)

print('Play Summary : Time taken = {0}, number of move = {1}, winner= {2}, Reason ={3}' .format(end-start, len(moves),winner,reason))