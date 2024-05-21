import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from flask import Flask, request, jsonify
import io
import argparse
import bitsandbytes as bnb
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)


class VisionChatBot:
    def __init__(self, model_path, precision='4bit'):

        from modules import models
        import modules.shared as shared
        shared.args.trust_remote_code = True
        shared.args.load_in_4bit = True

        self.model, self.tokenizer = models.load_model(
            model_path, 'Transformers')

        # Convert to the specified precision
        # self.model = self.model.to(dtype=self.torch_type)

        self.text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
        self.history = []
        self.image = None

    def set_image(self, image_file):
        try:
            image_stream = io.BytesIO(image_file.read())
            self.image = Image.open(image_stream).convert('RGB')
            logging.info("Image processed successfully")
        except Exception as e:
            logging.error(f"Failed to process image: {str(e)}")
            raise

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
        try:
            bot.set_image(request.files['image'])
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "Query text is required"}), 400
    try:
        response = bot.chat(query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ app.route('/clear', methods=['POST'])
def clear():
    logging.info("Clearing history")
    try:
        bot.clear_history()
        return jsonify({"message": "History cleared"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the Vision Chat Bot server")
    parser.add_argument('-m', '--model', type=str,
                        default="THUDM/cogvlm2-llama3-chinese-chat-19B", help="Path of the model to load")
    parser.add_argument('-ip', '--host', type=str, default='0.0.0.0',
                        help="Host IP address to run the server on")
    parser.add_argument('-port', '--port', type=int,
                        default=6000, help="Port to run the server on")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    bot = VisionChatBot(args.model)
    app.run(host=args.host, port=args.port)
