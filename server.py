import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from flask import Flask, request, jsonify
import io
import argparse
import bitsandbytes as bnb


app = Flask(__name__)


class MultiGPUModelLoader:
    def __init__(self, model_path, quantize_4bit=False, primary_gpu_mem_limit=20):
        self.model_path = model_path
        self.quantize_4bit = quantize_4bit
        self.primary_gpu_mem_limit = primary_gpu_mem_limit * 1e9  # Convert GB to bytes
        self.device0 = torch.device("cuda:0")
        self.device1 = torch.device("cuda:1")

    def load(self):
        config = AutoConfig.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_config(config)

        if self.quantize_4bit:
            # Apply 4-bit quantization using bitsandbytes
            model = bnb.nn.QuantLinear4bit.replace_all_linear(model)

        # Initially load the model to the primary GPU
        model = model.to(self.device0)
        torch.cuda.empty_cache()  # Clear any residual memory

        # Check if memory exceeds the limit
        if torch.cuda.memory_allocated(self.device0) > self.primary_gpu_mem_limit:
            # Split model layers if needed or move entirely
            self.split_and_allocate(model)

        return model

    def split_and_allocate(self, model):
        # This is a simplified version. You should implement this based on your model structure.
        num_layers = len(list(model.children()))
        split_point = num_layers // 2  # This is an arbitrary split point

        first_half = nn.Sequential(
            *list(model.children())[:split_point]).to(self.device0)
        second_half = nn.Sequential(
            *list(model.children())[split_point:]).to(self.device1)

        # Reconstruct the model by manually handling the forward pass or using nn.ModuleList
        self.model = nn.ModuleList([first_half, second_half])

    def forward(self, x):
        # Custom forward pass if you manually split layers
        x = self.model[0](x.to(self.device0))
        x = self.model[1](x.to(self.device1))
        return x


class VisionChatBot:
    def __init__(self, model_path, precision='4bit'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Set the data type based on precision
        if precision == 'float16':
            self.torch_type = torch.float16
        elif precision == 'bfloat16':
            self.torch_type = torch.bfloat16
        elif precision == 'int8':
            self.torch_type = torch.int8
        elif precision == '4bit':
            self.torch_type = torch.float16
        else:
            raise ValueError(
                "Unsupported precision type. Choose 'float16', 'bfloat16', 'int8', or '4bit'.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Load in float32 for potential quantization
            trust_remote_code=True,
        ).to(self.device).eval()

        if self.torch_type == torch.int8:
            # Apply quantization
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        elif self.torch_type == torch.float16 and precision == '4bit':
            bnb.optim.GlobalOptimManager.get_instance().override_config('stable', True)
            self.model = bnb.nn.QuantLinear4bit.replace_all_linear(self.model)

        # Convert to the specified precision
        self.model = self.model.to(dtype=self.torch_type)

        self.text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
        self.history = []
        self.image = None

    def set_image(self, image_file):
        if image_file:
            image_stream = io.BytesIO(image_file.read())
            self.image = Image.open(image_stream).convert('RGB')

    def chat(self, query):
        if self.image is None:
            if not self.history:
                query = self.text_only_template.format(query)
            else:
                old_prompt = ''
                for _, (old_query, response) in enumerate(self.history):
                    old_prompt += old_query + " " + response + "\n"
                query = old_prompt + "USER: {} ASSISTANT:".format(query)

        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=query,
            history=self.history,
            images=[self.image] if self.image is not None else None,
            template_version='chat'
        )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[input_by_model['images'][0].to(self.device).to(self.torch_type)]] if self.image is not None else None,
        }
        gen_kwargs = {"max_new_tokens": 2048, "pad_token_id": 128002}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
            # Note: This might need adjustment
            response = response.split("")[0]
        self.history.append((query, response))
        return response

    def clear_history(self):
        self.history = []


@ app.route('/chat', methods=['POST'])
def chat():
    if 'image' in request.files:
        image_file = request.files['image']
    else:
        image_file = None
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "Query text is required"}), 400
    bot.set_image(image_file)
    response = bot.chat(query)
    return jsonify({"response": response})


@ app.route('/clear', methods=['POST'])
def clear():
    bot.clear_history()
    return jsonify({"message": "History cleared"}), 200


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the Vision Chat Bot server")
    parser.add_argument('-m', '--model_path', type=str,
                        default="THUDM/cogvlm2-llama3-chinese-chat-19B", help="Path of the model to load")
    parser.add_argument('-ip', '--host', type=str, default='0.0.0.0',
                        help="Host IP address to run the server on")
    parser.add_argument('-port', '--port', type=int,
                        default=6000, help="Port to run the server on")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    bot = VisionChatBot(args.model_path)
    app.run(host=args.host, port=args.port)
