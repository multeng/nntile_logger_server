import asyncio
import datetime
import json
import os
import shutil
import subprocess

import tensorflow as tf

NODE_COUNTER = {}
BUS_COUNTER = {}

async def create_new_writer(log_dir):
    global writer
    current_time = datetime.datetime.now().strftime("%Y-%m-%d---%H%M")
    current_log_dir = os.path.join(log_dir, current_time)
    print(current_log_dir)
    os.makedirs(current_log_dir, exist_ok=True)
    if "writer" in globals():
        writer.close()

    writer = tf.summary.create_file_writer(current_log_dir)
    writer.set_as_default()
    return writer   

async def handle_new_logs(log_dir, split_hours):
    global writer
    global NODE_COUNTER
    global BUS_COUNTER
    NODE_COUNTER = {}
    BUS_COUNTER = {}
    if split_hours > 0:
        while True:
            writer = await create_new_writer(log_dir)
            # Convert hours into seconds
            await asyncio.sleep(split_hours * 60 * 60)
    else:
        writer = await create_new_writer(log_dir)

def increase_step(node, node_dict):
    if node not in node_dict:
        node_dict[node] = 1
    else:
        node_dict[node] = node_dict[node] + 1
        
async def start_tensorboard(log_dir):
    print(os.getcwd())
    process = await asyncio.create_subprocess_exec(
        'tensorboard',
        '--logdir',
        log_dir,
        '--reload_multifile=true',
        '--bind_all',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    print(f'TensorBoard stdout: {stdout.decode()}')
    print(f'TensorBoard stderr: {stderr.decode()}')        

def handle_flops_message(parsed_data):
    name = parsed_data.get("name")
    flops = float(parsed_data.get("flops"))
    increase_step(name, NODE_COUNTER)
    tf.summary.scalar(name, flops, NODE_COUNTER[name], name)

def handle_bus_message(parsed_data):
    bus_id = parsed_data.get("bus_id")
    total_bus_time = float(parsed_data.get("total_bus_time"))
    transferred_bytes = int(parsed_data.get("transferred_bytes"))
    transfer_count = int(parsed_data.get("transfer_count"))
    increase_step(bus_id, BUS_COUNTER)
    tf.summary.scalar(f"bus_{bus_id}/total_bus_time", total_bus_time, BUS_COUNTER[bus_id])
    tf.summary.scalar(f"bus_{bus_id}/transferred_bytes", transferred_bytes, BUS_COUNTER[bus_id])
    tf.summary.scalar(f"bus_{bus_id}/transfer_count", transfer_count, BUS_COUNTER[bus_id])

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"Connect from {addr}")
    while True:
        data = await reader.readline()
        if not data:
            break
        message = data.decode().strip()
        try:
            parsed_data = json.loads(message)
            print(parsed_data)
            message_type = int(parsed_data.get("type"))
            match message_type:
                case 0:
                    handle_flops_message(parsed_data)
                case 1:
                    handle_bus_message(parsed_data)
                case _:
                    print(f"Unknown message type: {message_type}")
        except json.JSONDecodeError:
            print("Error decoding JSON:", message)

async def main():
    log_dir = os.environ.get('LOG_DIR', 'logs')
    split_hours = int(os.environ.get('SPLIT_HOURS', 24))
    clear_logs = int(os.environ.get('CLEAR_LOGS', 1))
    server_port = int(os.environ.get('SERVER_PORT', 5001))

    os.makedirs(log_dir, exist_ok=True)
    print(f"log_dir={log_dir}, split_hours={split_hours}")
    server = await asyncio.start_server(
        handle_client, "0.0.0.0", server_port)

    addr = server.sockets[0].getsockname()
    print(f"Server has been started on {addr}")

    async def start_server():
        async with server:
            await server.serve_forever()

    if clear_logs:
        shutil.rmtree(log_dir)

    await asyncio.gather(
        handle_new_logs(log_dir, split_hours),
        start_server(),
        start_tensorboard(log_dir)
    )

if __name__ == '__main__':
    asyncio.run(main())
