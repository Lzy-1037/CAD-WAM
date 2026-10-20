"""Prompt / answer / state-string helpers of the text-state dataset WITHOUT the cadquery dependency.

Byte-identical to stepwise_dataset/build_text_state.py (INSTRUCTION, build_prompt, cap_state) and
stepwise_dataset/text_state.py (state_json, EMPTY_STATE, MAX_FACES), so `build` and `verify` run in the
verl env (pandas/pyarrow/transformers, no cadquery).  test_equivalence() asserts the equality wherever
the originals are importable (the cadquery env).
"""
import json

MAX_FACES = 40
EMPTY_STATE = {"solids": 0}

INSTRUCTION = (
    "You are constructing a CAD model step by step with cadseq version 1 actions. "
    "You are given the CAD description of the target model and the current state of the partially "
    "built model as a JSON summary of its solid: bounding box, volume, surface area, face/edge/vertex "
    "counts and the list of faces (type, center, normal or axis, radius, convexity, area), largest "
    "faces first, in the same coordinate frame and units as the actions; {\"solids\": 0} is the empty model. "
    "Generate ONLY one JSON object with two keys: \"action\", the next cadseq action preserving the "
    "described dimensions, and \"state\", the state summary of the model after that action in the same "
    "format as the current state. "
    "If the model is already complete, return {\"action\": {\"kind\": \"end\"}, \"state\": <the current state unchanged>}."
)
END_ACTION = '{"kind":"end"}'


def state_json(state) -> str:
    return json.dumps(state, separators=(",", ":"), ensure_ascii=False)


EMPTY_STATE_JSON = state_json(EMPTY_STATE)


def cap_state(state, max_faces=MAX_FACES):
    if "face_list" in state and len(state["face_list"]) > max_faces:
        state = dict(state, face_list=state["face_list"][:max_faces])
    return state


COUNT_LINE = "Actions completed so far: {}\n"       # identical wording to cad_eval/cad_prompt.py (latent v5)


def build_prompt(description: str, state: str, n_done=None) -> str:
    """n_done=None: the original text-state prompt; n_done=k: + the step-count line of the latent v5 prompt,
    placed like there (after the description, right before the state)."""
    extra = "" if n_done is None else COUNT_LINE.format(int(n_done))
    return f"{INSTRUCTION}\n\nCAD description:\n{description}\n\n{extra}Current state:\n{state}"


def build_answer(action: str, next_state: str) -> str:
    return '{"action":' + action + ',"state":' + next_state + "}"


def _round_sig(x, sig):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return [_round_sig(v, sig) for v in x] if isinstance(x, list) else (
            {k: _round_sig(v, sig) for k, v in x.items()} if isinstance(x, dict) else x)
    return float(f"{x:.{sig}g}") if isinstance(x, float) else x


def state_fingerprints(state: str):
    """(exact, coarse) 32-hex fingerprints of a state string.  `coarse` rounds every float to 3 significant
    digits first: OpenCASCADE's mass-property integration is not bit-reproducible, so a face centre can
    differ in the 5th digit at a rounding tie (measured: 8 of 4,479 states); real differences survive it."""
    import hashlib
    exact = hashlib.sha256(state.encode()).hexdigest()[:32]
    coarse = hashlib.sha256(state_json(_round_sig(json.loads(state), 3)).encode()).hexdigest()[:32]
    return exact, coarse


def test_equivalence():
    import build_text_state as b
    import text_state as t
    assert b.INSTRUCTION == INSTRUCTION and t.MAX_FACES == MAX_FACES and t.EMPTY_STATE == EMPTY_STATE
    s = {"solids": 1, "face_list": [{"area": float(i)} for i in range(50)], "faces": 50}
    assert b.cap_state(s, MAX_FACES) == cap_state(s) and t.state_json(s) == state_json(s)
    assert b.build_prompt("d", "{}") == build_prompt("d", "{}")
    return True
