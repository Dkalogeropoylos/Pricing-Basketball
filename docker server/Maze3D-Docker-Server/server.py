import time

from flask import Flask, request
from queue import Queue
from flask_cors import CORS
from environs import Env

env = Env()

agent = Queue(maxsize=1)
player = Queue(maxsize=1)
app = Flask(__name__)
CORS(app)
config_data = None
timeout = env('TIMEOUT', 10)
server_host = env('HOST', "https://maze-server.app.orbitsystems.gr")

def message_by(queue: Queue, data: dict):
    if queue.full():
        queue.get()
        # print(f"replacing {removed} with {data}")
    queue.put(data)


@app.route("/set_server_host", methods=["POST"])
def set_server_host():
    global server_host
    server_host = request.json['server_host']
    return {"OK": 'OK'}


@app.route("/env_variables")
def env_variables():
    return {"host": server_host}


@app.route("/config", methods=["POST", "GET"])
def config():
    global config_data
    if request.method == 'GET':
        print(config_data)
        return config_data
    if request.method == 'POST':
        config_data = request.json
        return {"OK": 'OK'}


@app.route("/")
def home():
    return "This is a http server for the maze experiment!!!"


@app.route("/agent_ready")
def agent_ready():
    print('Im searching for agent')
    """Initiate by agent"""
    print('agent->agent_ready')
    return assert_player_command(player.get(timeout=timeout), 'player_ready')


@app.route("/player_ready")
def player_ready():
    """Initiate by player"""
    print('player->player_ready')
    message_by(player, {'command': 'player_ready'})
    return agent.get(timeout=timeout)


@app.route("/reset")
def reset():
    """Initiate by agent"""
    #      print('agent->reset')
    message_by(agent, {'command': 'reset'})
    return assert_player_command(player.get(timeout=timeout), 'reset')


@app.route("/testreset")
def testreset():
    """Initiate by agent"""
    #      print('agent->reset')
    message_by(agent, {'command': 'testreset'})
    return assert_player_command(player.get(timeout=timeout), 'reset')

@app.route("/reset_done", methods=["POST"])
def reset_done():
    """Initiate by player"""
    #      print('player->reset_done')
    message_by(player, request.json)
    return agent.get(timeout=timeout)


@app.route("/step", methods=["POST"])
def step():
    """Initiate by agent"""
    message_by(agent, {'command': 'step', "step_request": request.json})
    return assert_player_command(player.get(timeout=timeout), 'step')


@app.route("/step_two_agents", methods=["POST"])
def step_two_agents():
    """Initiate by agent"""
    message_by(agent, {'command': 'step_two_agents', "step_request": request.json})
    return assert_player_command(player.get(timeout=timeout), 'step')


@app.route("/observation", methods=["POST"])
def observation():
    """Initiate by player"""
    print('player->observation')
    message_by(player, request.json)
    return agent.get(timeout=timeout)


@app.route("/training", methods=["POST"])
def training():
    """Initiate by agent"""
    # print('agent->training', request.json)
    message_by(agent, {'command': 'training', "training_request": request.json})
    return {"OK": 'OK'}


def assert_player_command(res, command):
    if 'command' in res and res['command'] == command:
        return res
    # print("ERROR COMMAND", res, command)
    return player.get(timeout=timeout)


@app.route("/finished")
def finished():
    """Initiate by agent"""
    print('agent->finished')
    message_by(agent, {'command': 'finished'})
    return {"OK": 'OK'}


if __name__ == "__main__":
    #print('######################## IMAGONAWORKNOW #########################')

    app.run(port=8080, host="0.0.0.0")
