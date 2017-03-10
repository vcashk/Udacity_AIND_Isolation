"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player,x=1,y=0.5,z=1):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """


    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(x * own_moves - y * opp_moves + z)


    #raise NotImplementedError

def eval_param_score_fn(game, player,x=1,y=0.5,z=1):
    """Calculate the heuristic value of a game state from the point of view
    of the given player. Hyper parameterized evaluation function
     Here x,y,z are tuning parameter.
        x range from 0 to 1
        y range from 0 to 1
        z is a scalar.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """


    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(x*own_moves - y*opp_moves + z )

def eval_reduce_opp_score_fn(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player. Evaluate based on opponents score.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """


    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(-opp_moves)

def eval_normalize_score_fn(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player. The function evaluates based on the number of occupied space.
    When board positions are less than half occupied, agent weighs to reduce opponent's legal move. (Aggressive play)


    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """


    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    own_moves = len(game.get_legal_moves(player))
    print (len(game.get_blank_spaces()))
    if len(game.get_blank_spaces())>25:
        return  float(0.5*own_moves-opp_moves)
    else:
        return float(own_moves - 0.5*opp_moves)



class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='alphabeta', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left



        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        current_best_move = None

        # opening moves

        #If first move  - occupy central move
        #if len(game.get_blank_spaces())==49:
        #    return (3,3)
        #if player2 and center not occupied
        #if len(game.get_blank_spaces())==48 and (3,3) in legal_moves:
        #    return (3,3)


        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
             depth = 1
             #maximum possible iteration or depth
             #max_depth = len(game.get_blank_spaces())
             #iternative deeping
             while True:
             #while depth <= max_depth:
                 if self.method == 'minimax':
                     _, current_best_move = self.minimax(game, depth)
                 else:
                     _, current_best_move=self.alphabeta(game, depth)

                 depth += 1
                 #print(current_best_move)
                 #print(depth)
            else:
                 if self.method == 'minimax':
                     _, current_best_move = self.minimax(game, self.search_depth)
                 else:
                     _, current_best_move=self.alphabeta(game, self.search_depth)

        except Timeout:

                # Handle any actions required at timeout, if necessary
             #  print("timeout - log check")
                #print(current_best_move)
                #print(self.time_left())
                return current_best_move


        # Return the best move from the last completed search iteration
        return current_best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """

        # when the bottom of the depth called  - when calling the function recursivly
        if depth == 0:
            return self.score(game, self), (-1, -1)
        if game.utility(self)!= 0.0:
            return game.utility(self), (-1, -1)
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        #initialize varibale and get legal moves for root node

        legal_moves=game.get_legal_moves()

        # random order of the node to see different result
        #legal_moves=random.shuffle(game.get_legal_moves())
        #legal_moves= random.shuffle(legal_moves))
        best_move = legal_moves[0]
        if maximizing_player:
            # intialize the  worst possible value for value for maximizer - infinity

            best_value=float("-inf")

            # for each child node
            for  move in legal_moves:
                child_game = game.forecast_move(move)
                value,_= self.minimax(child_game, depth-1, maximizing_player=False)
                #print(value,move)
                if best_value<value:
                    best_value=value
                    best_move=move
            return best_value,best_move

        else:
            # intialize the worst possible value for value for minimizer + infinity
            best_value=float("inf")
            # for each child node
            for  move in legal_moves:
                child_game = game.forecast_move(move)
                value,_= self.minimax(child_game,  depth-1, maximizing_player=True)
                #print(value, move)
                if best_value > value:
                    best_value = value
                    best_move = move
            return best_value, best_move




    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
             return self.score(game, self), (-1, -1)
        if game.utility(self) != 0.0:
            return game.utility(self), (-1, -1)
        legal_moves=game.get_legal_moves()
        best_move = legal_moves[0]
        if maximizing_player:
            # for each child node
            for move in legal_moves:
                child_game = game.forecast_move(move)
                #get alpha value of child  -  child node is minimizing node
                value, _ = self.alphabeta(child_game, depth - 1,alpha,beta, maximizing_player=False)
                #print(alpha,value, move)
                #max(value,alpha)
                if value > alpha:
                    alpha = value
                    best_move = move
                # prune when alpha>= beta
                if alpha >= beta:
                    break

            return alpha,best_move

        else:
            # for each child node
            for move in legal_moves:
                child_game = game.forecast_move(move)
                # get beta value of child  -  child node is maximizing node
                value, _ = self.alphabeta(child_game, depth - 1, alpha, beta, maximizing_player=True)
                # min(value,beta)

                if value < beta:
                    beta = value
                    best_move = move

                # prune when alpha>= beta

                if alpha >= beta:
                    break

            return beta, best_move


