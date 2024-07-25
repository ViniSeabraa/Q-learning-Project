import connection as cn
import socket
import numpy as np
import random

def establish_connection(port):
    conn = cn.connect(port)
    if isinstance(conn, socket.socket):
        print("Connection successful.")
        return conn
    else:
        print("Connection failed.")
        return None

def initialize_q_table(file_path, shape):
    try:
        q_table = np.loadtxt(file_path)
        print("Q-table loaded successfully.")
    except Exception as e:
        print(f"Error loading Q-table: {e}. Initializing new Q-table with zeros.")
        q_table = np.zeros(shape)
    return q_table

def persist_q_table(q_table, file_path):
    np.savetxt(file_path, q_table)
    print("Q-table saved.")

def select_action(state_index, q_table, epsilon, action_list):
    if random.uniform(0, 1) < epsilon:
        chosen_action = random.choice(action_list)
        print(f"Random action selected for state {state_index}: {chosen_action}")
    else:
        action_index = np.argmax(q_table[state_index])
        chosen_action = action_list[action_index]
        print(f"Optimal action selected for state {state_index}: {chosen_action}")
    return chosen_action

def update_q_value(q_table, current_state, action_index, reward, next_state, learning_rate, discount_factor):
    max_future_q = np.max(q_table[next_state])
    old_q = q_table[current_state, action_index]
    new_q = (1 - learning_rate) * old_q + learning_rate * (reward + discount_factor * max_future_q)
    q_table[current_state, action_index] = new_q
    print(f"Updated Q-value for state {current_state}, action {action_index}: {new_q}")

def extract_state_details(state):
    platform = int(state[2:7], 2)
    direction = int(state[-2:], 2)
    return platform, direction

def main():
    port = 2037
    q_table_shape = (24 * 4, 3)
    file_path = 'result.txt'

    learning_rate = 0.7
    discount_factor = 0.95
    exploration_rate = 0.1

    actions = ["left", "right", "jump"]
    action_map = {action: idx for idx, action in enumerate(actions)}

    connection = establish_connection(port)
    if not connection:
        return

    q_table = initialize_q_table(file_path, q_table_shape)

    initial_state, initial_reward = cn.get_state_reward(connection, "")
    platform, direction = extract_state_details(initial_state)
    print(f"Initial state: {initial_state}, Platform: {platform}, Direction: {direction}, Reward: {initial_reward}")

    current_state_index = int(initial_state, 2)

    while True:
        selected_action = select_action(current_state_index, q_table, exploration_rate, actions)
        action_idx = action_map[selected_action]

        next_state, reward = cn.get_state_reward(connection, selected_action)
        platform, direction = extract_state_details(next_state)
        print(f"Action: {selected_action}, Next state: {next_state}, Platform: {platform}, Direction: {direction}, Reward: {reward}")

        next_state_index = int(next_state, 2)
        update_q_value(q_table, current_state_index, action_idx, reward, next_state_index, learning_rate, discount_factor)

        current_state_index = next_state_index
        persist_q_table(q_table, file_path)

main()