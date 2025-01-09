import json, os, tqdm
import jittor as jt

from JDiffusion.pipelines import StableDiffusionPipeline

max_num = 15
dataset_root = "./A"
prompt_file = "prompt"

with jt.no_grad():
    for tempid in tqdm.tqdm(range(0, max_num)):
        taskid = "{:0>2d}".format(tempid)
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
        pipe.load_lora_weights(f"style/style_{taskid}")

        # load json
        with open(f"{dataset_root}/{taskid}/{prompt_file}.json", "r") as file:
            prompts = json.load(file)

        for id, prompt in prompts.items():
            # print(prompt)
            # myprompt = prompt + f" in style_{taskid}" # 0
            # myprompt = prompt + f", style_{taskid}" # 0.1
            # myprompt = "an artwork in style sks depicting " + prompt.lower() + " with material and texture of style sks" # 1
            # myprompt = "an artwork in style sks depicting " + prompt.lower() + " with material and texture of style rcn" # 1.1
            # myprompt = "one " + prompt.lower() + " depicted in artwork style sks" # 1.2
            # myprompt = prompt.lower() + ". Depicted in style rcn." # 2
            myprompt = "nlwx " +  prompt.lower() # 2.1
            image = pipe(myprompt, num_inference_steps=25, width=512, height=512).images[0]
            os.makedirs(f"./output/{taskid}", exist_ok=True)
            image.save(f"./output/{taskid}/{prompt}.png")