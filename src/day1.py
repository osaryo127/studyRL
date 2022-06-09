from enum import Enum
from typing import List, Dict, Tuple
import numpy as np
import random


class State:
    def __init__(self, row: int=-1, column: int=-1):
        self.row = row
        self.column = column
    
    def __repr__(self):
        return f"State: [{self.row}, {self.column}]"
    
    def clone(self):
        return State(self.row, self.column)
    
    def __hash__(self):
        return hash((self.row, self.column))
    
    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment:
    def __init__(self, grid: List[List[int]], move_prob: float=0):
        """
        エージェントがゴールを早く目指すように、デフォルトの報酬を負値に定義。
        エージェントは選択した向きにmove_probの確率で移動し、(1 - move_prob)の確率で他の方向に進む。
        """
        # grid is 2d-Array. Its values are treated as an attribute.
        # kinds of attributes is following
        # 0: ordinary cell
        # -1: damage cell (game end)
        # 1: reward cepp (game end)
        # 9: block cell (can't locate agent)
        self.grid = grid
        self.agent_state = State()
        self.default_reward = -0.04
        self.move_prob = move_prob
        self.reset()
        
    @property
    def row_length(self) -> int:
        return len(self.grid)
    
    @property
    def column_length(self) -> int:
        return len(self.grid[0])
    
    @property
    def actions(self) -> List[Action]:
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    
    @property
    def states(self) -> List[State]:
        """
        迷路内の移動可能なセルを返す。(ブロックセルを除外)
        """
        states = []
        for row in range(self.row_length):
            for col in range(self.column_lengthh):
                if self.grid[row][col] != 9:
                    states.append(State(row, col))
        return states

    def transition_func(self, state: State, action: Action) -> Dict[State, float]:
        """
        状態とアクションを受けとり、次の状態への遷移確率を返す。
        今回の迷路では、move_probの確率で選択した方向に、
        (1-move_prob)の確率で、選択した方向との反対以外の方向に等確率で遷移する。
        """
        transition_probs = {}
        if not self.can_action_at(state):
            # Already on the terminal cell.
            return transition_probs
        
        opposite_direction = Action(action.value * -1)
        
        for suggest_action in self.actions:
            next_state = self._move(state, suggest_action)
            
            if suggest_action == action:
                prob = self.move_prob
            elif suggest_action != opposite_direction :
                prob = (1 - self.move_prob) / 2
            else:
                prob = 0
                
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs
    
    def can_action_at(self, state: State) -> bool:
        """
        stateがアクション可能なセルかどうか判定
        """
        # Indexエラー（迷路外の座標を参照）をキャッチするように実装
        try:
            if self.grid[state.row][state.column] == 0:
                return True
            else:
                return False
        except IndexError as e:
            raise ValueError(f"This state is out of mase! {state}")
    
    def _move(self, state: State, action) -> State:
        """
        位置とアクションを受け取り、アクション可能な位置であれば、
        受け取ったアクション方向に移動した位置に移動する。
        移動先位置が迷路の外であれば、そのままの位置を返す。
        """
        if not self.can_action_at:
            raise ValueError("Can't move from here!")
        
        next_state = state.clone()

        # Execute an action (move).
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # Check whether a state is out of the grid.
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # Check whether the agent bumped a block cell.
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state: State) -> Tuple[float, bool]:
        """
        受け取ったstateの報酬と、ゲームが終了したか否かを返す
        デフォルトでは負の値（-0.04）を設定しているが、
        歩き回るだけでは報酬が減る（早くゴールするよう促す）影響を与える。
        """
        reward = self.default_reward
        done = False

        # Check an attribute of next state.
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            # Get reward! and the game ends.
            reward = 1
            done = True
        elif attribute == -1:
            # Get damage! and the game ends.
            reward = -1
            done = True

        return reward, done

    def reset(self) -> State:
        # Locate the agent at lower left corner.
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action) -> Tuple[State, float, bool]:
        """
        現在のエージェントの状態にアクションを適用し、遷移先状態, 即時報酬, 終了判定を返す
        """
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state: State, action: Action) -> Tuple[State, float, bool]:
        """
        状態とアクションを受け取り、遷移関数によって遷移確率を算出、。
        確率に応じた繊維を実行し、遷移先状態、即時報酬、終了判定を返す。
        """
        transition_probs = self.transition_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done


class Agent():
    
    def __init__(self, env: Environment):
        """
        エージェントの初期化
        """
        self.actions = env.actions
        
    def policy(self, state: State) -> Action:
        """
        状態を受け取り、アクションを決定する。
        """
        return random.choice(self.actions)