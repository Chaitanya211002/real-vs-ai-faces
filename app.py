import os, json, random
import numpy as np
import gradio as gr
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.tflite")
IMG_DIR = os.path.join(BASE_DIR, "test_images")
MODEL_NAME = "MobileNetV3Small (transfer learning, frozen backbone)"
PIXEL_SCALE = 1.0      
DISPLAY_SIZE = 384    
LABEL_TXT = {1: "Real", 0: "AI-generated"}

# Load model and images once at startup
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
IN_DET = interpreter.get_input_details()[0]
OUT_DET = interpreter.get_output_details()[0]

def model_predict(img_uint8):
    x = img_uint8[np.newaxis].astype(np.float32) * PIXEL_SCALE
    interpreter.set_tensor(IN_DET["index"], x)
    interpreter.invoke()
    return float(interpreter.get_tensor(OUT_DET["index"])[0, 0])

with open(os.path.join(IMG_DIR, "labels.json")) as f:
    LABELS = json.load(f)
NAMES = sorted(LABELS)
IMAGES = np.stack([np.array(Image.open(os.path.join(IMG_DIR, n)).convert("RGB").resize((128, 128)))
                   for n in NAMES])
Y_TRUE = np.array([LABELS[n] for n in NAMES])

MODEL_SCORES = np.array([model_predict(img) for img in IMAGES])
MODEL_PREDS = (MODEL_SCORES > 0.5).astype(int)
MODEL_ACC = float(np.mean(MODEL_PREDS == Y_TRUE))
print(f"Loaded {len(NAMES)} images | model accuracy on them: {MODEL_ACC:.3f}")

def display_image(idx):
    return np.array(Image.fromarray(IMAGES[idx]).resize((DISPLAY_SIZE, DISPLAY_SIZE),
                                                        Image.Resampling.BICUBIC))

def card(title, score, answered, color):
    acc = f"{score / answered:.0%}" if answered else "–"
    return (f"<div style='flex:1;padding:14px;border-radius:12px;background:{color};"
            f"color:white;text-align:center'>"
            f"<div style='font-size:15px;text-transform:uppercase;letter-spacing:1px'>{title}</div>"
            f"<div style='font-size:30px;font-weight:700'>{score} / {answered}</div>"
            f"<div style='font-size:14px'>Accuracy: {acc}</div></div>")

def scoreboard(state):
    answered = state["round"] + (1 if state["answered"] else 0)
    return (f"<div style='display:flex;gap:12px;margin-bottom:12px'>"
            f"{card('You', state['human'], answered, '#4C72B0')}"
            f"{card('Model', state['model'], answered, '#DD8452')}</div>")

def round_text(state):
    return f"### Round {min(state['round'] + 1, state['n'])} of {state['n']}"

def badge(ok):
    color = "#2e9e5b" if ok else "#d64541"
    text = "Correct" if ok else "Incorrect"
    return (f"<span style='display:inline-block;margin-left:6px;padding:1px 8px;border-radius:10px;"
            f"background:{color};color:white;font-size:12px;font-weight:600'>{text}</span>")

def feedback_html(idx, user_label):
    truth, m_pred = int(Y_TRUE[idx]), int(MODEL_PREDS[idx])
    score = float(MODEL_SCORES[idx])
    conf = score if m_pred == 1 else 1 - score
    human_ok, model_ok = user_label == truth, m_pred == truth
    bg, border = ("#e8f7ee", "#2e9e5b") if human_ok else ("#fdecea", "#d64541")
    return f"""<div style='padding:16px;border-radius:12px;background:{bg};border:3px solid {border};color:#222'>
      <div style='font-size:22px;font-weight:700'>{'You got it!' if human_ok else 'Not quite!'}</div>
      <div style='margin-top:8px'>This face is <b>{LABEL_TXT[truth]}</b></div>
      <div style='margin-top:4px'>You said: <b>{LABEL_TXT[user_label]}</b>{badge(human_ok)}</div>
      <div style='margin-top:4px'>Model said: <b>{LABEL_TXT[m_pred]}</b> ({conf:.0%} confident){badge(model_ok)}</div>
    </div>"""

CONFETTI_CSS = """<style>
@keyframes fall { 0% {transform: translateY(-30px) rotate(0deg); opacity: 1;}
                  100% {transform: translateY(260px) rotate(540deg); opacity: 0;} }
.confetti {position: absolute; inset: 0; pointer-events: none;}
.confetti span {position: absolute; top: 0; width: 8px; height: 14px; border-radius: 2px;
                animation: fall 2.8s ease-in infinite;}
</style>"""
CONFETTI_COLORS = ["#FFD166", "#EF476F", "#06D6A0", "#118AB2", "#FFFFFF"]

def final_html(state):
    h, m, n = state["human"], state["model"], state["n"]
    if h > m:
        title, bg, party = "You beat the AI detector!", "linear-gradient(135deg,#2e9e5b,#7bd389)", True
    elif h < m:
        title, bg, party = "The machine wins this time!", "linear-gradient(135deg,#d64541,#f39c6b)", False
    else:
        title, bg, party = "It's a tie!", "linear-gradient(135deg,#4C72B0,#8fb3e8)", False
    confetti = ""
    if party:
        pieces = "".join(f"<span style='left:{random.randint(2, 95)}%;"
                         f"animation-delay:{random.random() * 2:.2f}s;"
                         f"background:{random.choice(CONFETTI_COLORS)}'></span>" for _ in range(30))
        confetti = CONFETTI_CSS + f"<div class='confetti'>{pieces}</div>"
    return f"""<div style='position:relative;overflow:hidden;padding:28px 16px;border-radius:14px;
                background:{bg};color:white;text-align:center;min-height:200px'>
      {confetti}
      <div style='font-size:28px;font-weight:800'>{title}</div>
      <div style='font-size:18px;margin-top:10px'>You: {h}/{n} ({h / n:.0%}) &nbsp;|&nbsp; Model: {m}/{n} ({m / n:.0%})</div>
      <div style='font-size:14px;margin-top:10px;opacity:.9'>On the full {len(NAMES)}-image test set, the model scores
      {MODEL_ACC:.0%}. Press "Start new game" to play again!</div>
    </div>"""

def show_current(state):
    idx = state["order"][state["round"]]
    return (state, display_image(idx), round_text(state),
            "<div style='padding:16px;color:#666'>Is this face real or AI-generated? Make your guess!</div>",
            scoreboard(state),
            gr.update(interactive=True), gr.update(interactive=True),
            gr.update(interactive=False, value="Next image"))

def new_game(n_rounds):
    n = int(n_rounds)
    real_idx = np.where(Y_TRUE == 1)[0].tolist()
    ai_idx = np.where(Y_TRUE == 0)[0].tolist()
    n_real = n // 2 + random.randint(0, n % 2)             
    order = random.sample(real_idx, n_real) + random.sample(ai_idx, n - n_real)
    random.shuffle(order)
    state = {"order": order, "round": 0, "n": n, "human": 0, "model": 0, "answered": False}
    return show_current(state)

def make_guess(state, user_label):
    if state is None or state["answered"] or state["round"] >= state["n"]:
        return (state,) + (gr.update(),) * 7                
    idx = state["order"][state["round"]]
    state["human"] += int(user_label == Y_TRUE[idx])
    state["model"] += int(MODEL_PREDS[idx] == Y_TRUE[idx])
    state["answered"] = True
    last = state["round"] == state["n"] - 1
    return (state, gr.update(), round_text(state), feedback_html(idx, user_label), scoreboard(state),
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=True, value="See final results" if last else "Next image"))

def next_round(state):
    if state is None or not state["answered"]:
        return (state,) + (gr.update(),) * 7
    state["round"] += 1
    state["answered"] = False
    if state["round"] >= state["n"]:
        return (state, gr.update(), "### Game over!", final_html(state), scoreboard(state),
                gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))
    return show_current(state)

# Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Real or AI? Human vs Model") as demo:
    gr.Markdown(f"""# Human vs. Model: Real or AI-Generated?
Can you spot AI-generated faces better than a neural network? Each round shows one face from our held-out test set.
Guess **Real** or **AI-generated**, then see whether you, and the model, got it right.""")

    state = gr.State(None)
    with gr.Row():
        n_rounds = gr.Radio([5, 10, 20], value=10, label="Rounds per game")
        start_btn = gr.Button("Start new game", variant="primary")

    with gr.Row():
        with gr.Column(scale=3):
            round_md = gr.Markdown()
            image = gr.Image(show_label=False, interactive=False, height=DISPLAY_SIZE)
            with gr.Row():
                real_btn = gr.Button("Real", size="lg")
                ai_btn = gr.Button("AI-generated", size="lg")
            next_btn = gr.Button("Next image", interactive=False)
        with gr.Column(scale=2):
            score_box = gr.HTML()
            feedback_box = gr.HTML()

    gr.Markdown("Built for a deep learning course assignment: MLP → CNN → augmentation → transfer learning → fine-tuning.")

    outputs = [state, image, round_md, feedback_box, score_box, real_btn, ai_btn, next_btn]
    start_btn.click(new_game, inputs=n_rounds, outputs=outputs)
    real_btn.click(lambda s: make_guess(s, 1), inputs=state, outputs=outputs)
    ai_btn.click(lambda s: make_guess(s, 0), inputs=state, outputs=outputs)
    next_btn.click(next_round, inputs=state, outputs=outputs)
    demo.load(new_game, inputs=n_rounds, outputs=outputs)   # start a game when the page opens

if __name__ == "__main__":
    # 0.0.0.0 + $PORT is required on Render; locally this falls back to port 7860
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
