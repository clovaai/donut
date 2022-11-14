"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse

import gradio as gr
import torch
from PIL import Image

from donut import DonutModel


def demo_process_vqa(input_img, question):
    global pretrained_model, task_prompt, task_name
    input_img = Image.fromarray(input_img)
    user_prompt = task_prompt.replace("{user_input}", question)
    output = pretrained_model.inference(input_img, prompt=user_prompt)["predictions"][0]
    return output


def demo_process(input_img):
    global pretrained_model, task_prompt, task_name
    input_img = Image.fromarray(input_img)
    output = pretrained_model.inference(image=input_img, prompt=task_prompt)["predictions"][0]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="docvqa")
    parser.add_argument("--pretrained_path", type=str, default="naver-clova-ix/donut-base-finetuned-docvqa")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--sample_img_path", type=str)
    args, left_argv = parser.parse_known_args()

    task_name = args.task
    if "docvqa" == task_name:
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    else:  # rvlcdip, cord, ...
        task_prompt = f"<s_{task_name}>"

    example_sample = []
    if args.sample_img_path:
        example_sample.append(args.sample_img_path)

    pretrained_model = DonutModel.from_pretrained(args.pretrained_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        device = torch.device("cuda")
        pretrained_model.to(device)

    pretrained_model.eval()

    demo = gr.Interface(
        fn=demo_process_vqa if task_name == "docvqa" else demo_process,
        inputs=["image", "text"] if task_name == "docvqa" else "image",
        outputs="json",
        title=f"Donut 🍩 demonstration for `{task_name}` task",
        examples=[example_sample] if example_sample else None,
    )
    demo.launch(server_name=args.url, server_port=args.port)
