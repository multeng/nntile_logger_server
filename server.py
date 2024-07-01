import asyncio
import os
import subprocess
import datetime
import shutil
import asyncio
import json
import tensorflow as tf
from websockets.server import serve


is_split = True
log_dir = "logs/"
delay_for_log = 60 * 60
NODE_COUNTER = {}

def increase_step(node, node_dict):
    if node not in node_dict:
        node_dict[node] = 1
    else:
        node_dict[node] += 1


async def create_new_writer():
    global writer
    current_time = datetime.datetime.now().strftime("%Y-%m-%d---%H%M")
    current_log_dir = os.path.join(log_dir, current_time)
    print(f"Creating new log directory: {current_log_dir}")
    os.makedirs(current_log_dir, exist_ok=True)
    if "writer" in globals():
        writer.close()
    
    writer = tf.summary.create_file_writer(current_log_dir)
    writer.set_as_default()
    return writer

async def handle_new_logs():
    global writer
    global NODE_COUNTER
    NODE_COUNTER = {}
    if is_split:
        while True:
            writer = await create_new_writer()
            await asyncio.sleep(delay_for_log)
    else:
        writer = await create_new_writer()


async def start_tensorboard():
    print(os.getcwd())
    process = await asyncio.create_subprocess_exec(
        'tensorboard', 
        '--logdir', log_dir,
        '--reload_multifile=true',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    print(f'TensorBoard stdout: {stdout.decode()}')
    print(f'TensorBoard stderr: {stderr.decode()}')

async def handler(websocket):
    while True:
        message = await websocket.recv()
        try:
            parsed_data = json.loads(message)
            name = parsed_data.get("name")
            flops = float(parsed_data.get("flops"))
            increase_step(name, NODE_COUNTER)
            tf.summary.scalar(name, flops, NODE_COUNTER[name])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {message}")


async def main():
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        shutil.rmtree(log_dir)
    except Exception as e:
        print(f"Error removing log directory: {e}")

    async def start_server():
        async with serve(handler, "localhost", 5001):
            await asyncio.Future()

    await asyncio.gather(
        handle_new_logs(),
        start_server(),
        start_tensorboard()
    )


if __name__ == "__main__":
    asyncio.run(main())