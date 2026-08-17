# from llava.constants import (DEFAULT_IM_END_TOKEN,
#                              DEFAULT_IM_START_TOKEN,
#                              IMAGE_TOKEN_INDEX,
#                              DEFAULT_IMAGE_PATCH_TOKEN,
#                              DEFAULT_IMAGE_TOKEN)


# the image start and end token is not used
MOTION_TOKEN_INDEX = -300
DEFAULT_MOTION_TOKEN = "<motion>"
# DEFAULT_MOTION_TOKEN = "[SEG]"
# DEFAULT_MOTION_PATCH_TOKEN = "<motion_patch>"
# DEFAULT_MOTION_START_TOKEN = "<motion_start>"
# DEFAULT_MOTION_END_TOKEN = "<motion_end>"
# MOTION_PLACEHOLDER = "<motion-placeholder>"

HMR_SHORT_QUESTION_LIST = [
    "I have a description of a person's pose, can you give the SMPL pose of this person?",
    "Give you a word descrption of a human, please output the SMPL pose.",
    "Describe what this perosn is doing using SMPL pose.",
    "What's the SMPL pose of this person?",
    "Use SMPL pose to describe this person's behavior."
]

# TEXT_SHORT_QUESTION_LIST = [
#     "Can you give the SMPL pose?",
#     "Please output this person's SMPL pose.",
#     "Give the SMPL pose.",
#     "What's the SMPL pose of it?",
#     "Use SMPL to describe the pose."
# ]

# for text to smpl, {sent} might indicate the sentence
TEXT_SHORT_QUESTION_LIST = [
    "I have a word description of a person's pose, can you give the SMPL pose of this person? {sent}",
    "There is a person: {sent} Please output this person's SMPL pose.",
    "{sent} Give the SMPL pose.",
    "What's the SMPL pose of this person? {sent}",
    "Use SMPL pose to describe this person's behavior. {sent}",
    "There is a person doing this: {sent} Can you use SMPL pose to describe the pose?",
    "A person is described as: {sent} Use the SMPL pose to reflect this.",
    "Human pose is described as words: {sent} The SMPL pose is?",
    "Human pose can be described as words: {sent} And it can also be described as SMPL pose format, can you output this?",
]

MOTION_TO_TEXT_QUESTION_LIST = [
    DEFAULT_MOTION_TOKEN + "\n" + "Can you describe the motion of the person?",
    DEFAULT_MOTION_TOKEN
    + "\n"
    + "What is the human motion? Please respond with text description.",
    DEFAULT_MOTION_TOKEN
    + "\n"
    + "What is the person doing? Please describe in detail.",
]
from llava.constants import DEFAULT_IMAGE_TOKEN

IMAGE_TO_TEXT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you describe the motion of the person?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is the human motion? Please respond with text description.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is the person doing? Please describe in detail.",
]

IMAGE_MOTION_TO_TEXT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_MOTION_TOKEN + "\n" + "Can you describe the motion of the person?",
    DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_MOTION_TOKEN + "\n"
    + "What is the human motion? Please respond with text description.",
    DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_MOTION_TOKEN + "\n"
    + "What is the person doing? Please describe in detail.",
]

MOTION_TO_MOTION_QUESTION_LIST = [
    DEFAULT_MOTION_TOKEN + "\n" + "Can you predict the motion sequence of the person?",
    DEFAULT_MOTION_TOKEN + "\n" + "What is the person doing? Please respond with motion sequence.",
    DEFAULT_MOTION_TOKEN + "\n" + "What is the human motion? Please respond with motion sequence.",
]

MOTION_TEXT_TO_MOTION_QUESTION_LIST = [
    DEFAULT_MOTION_TOKEN + "\n" + "Can you predict the motion sequence of the person considering the following description?",
    DEFAULT_MOTION_TOKEN + "\n" + "What is the person doing? Please respond with motion sequence based on the following description.",
    DEFAULT_MOTION_TOKEN + "\n" + "What is the human motion? Please respond with motion sequence based on the following description.",
]

TEXT_TO_MOTION_QUESTION_LIST = [
    "Can you predict the motion with the following description?",
    "Please predict the motion sequence based on the following description.",
    "What is the motion sequence of the following description?",
    "What is the motion sequence of the following text?",
]

MOTION_ANSWER_LIST = [
    f"It is {DEFAULT_MOTION_TOKEN}.",
    f"Sure, {DEFAULT_MOTION_TOKEN}.",
    f"Sure, it is {DEFAULT_MOTION_TOKEN}.",
    f"Sure, the motion sequence is {DEFAULT_MOTION_TOKEN}.",
]

MOTION_TOKEN_LIST = []
motion_token_num = 128
for i in range(motion_token_num):
    MOTION_TOKEN_LIST.append(f"[MOTION_{i}]")


# ===========================================================================
# Nymeria per-caption-type questions (4-category training and evaluation)
# ===========================================================================
# The lists above are the release's, written when only atomic actions existed.
# With four caption types the question has to say which one it asks for: three
# of the categories are annotated on the SAME windows (1,371 test windows carry
# all four, 8,373 carry the three body-part ones), so the motion and the image
# are identical and only the wording distinguishes the targets. Each question
# below therefore names its own category -- "action"/"step by step" for atomic,
# "hand"/"arm", "leg"/"foot", "posture"/"stance".
#
# Structure follows the release exactly: a question body per category, composed
# over the same three modality prefixes, sampled once per sample at build time.

CAPTION_TYPE_ATOMIC = "Describe my atomic actions"
CAPTION_TYPE_HANDS = "Describe my hands/arms motion"
CAPTION_TYPE_LEGS = "Describe my legs/feet motion"
CAPTION_TYPE_POSTURE = "Describe my body posture"

NYMERIA_CAPTION_TYPES = [
    CAPTION_TYPE_ATOMIC,
    CAPTION_TYPE_HANDS,
    CAPTION_TYPE_LEGS,
    CAPTION_TYPE_POSTURE,
]

CATEGORY_QUESTION_BODIES = {
    CAPTION_TYPE_ATOMIC: [
        "Can you describe the motion of the person?",   # <- used at evaluation
        "Can you describe the person's actions step by step?",
        "Describe the sequence of actions the person performs.",
        "What actions does the person carry out here? Please respond with a text description.",
        "Narrate the actions of the person, including the objects they interact with.",
        "Walk me through the person's actions one step at a time.",
        "Report the person's actions and object interactions.",
        "Give a detailed account of the actions the person performs.",
        "Please describe the person's actions in detail.",
    ],
    CAPTION_TYPE_HANDS: [
        "In the context of the scene, describe the hand and arm motion in plain text.",  # <- eval
        "Can you describe the movement of the person's hands and arms?",
        "What are the person's arms doing? Please respond with a text description.",
        "Describe the hand and arm movement of the person in detail.",
        "How does the person move their hands and arms here?",
        "Report what the person does with their hands and arms.",
        "Describe the person's arm and hand movement, including anything they hold.",
        "What hand and arm movement do you observe? Answer in words.",
        "Give a detailed account of the person's hand and arm movement.",
    ],
    CAPTION_TYPE_LEGS: [
        "Observing the area, describe the leg and foot motion accurately.",  # <- eval
        "Can you describe the movement of the person's legs and feet?",
        "What are the person's legs and feet doing? Please respond with a text description.",
        "Describe the leg and foot movement of the person in detail.",
        "How does the person move their legs and feet here?",
        "Report the person's stepping and foot placement.",
        "Describe the person's leg and foot movement.",
        "What leg and foot movement do you observe? Answer in words.",
        "Give a detailed account of the person's leg and foot movement.",
    ],
    CAPTION_TYPE_POSTURE: [
        "Based on the spatial context, describe the body posture clearly.",  # <- eval
        "Can you describe the person's body posture?",
        "How is the person's body positioned? Please respond with a text description.",
        "Describe the body posture and pose of the person in detail.",
        "Is the person standing, sitting, leaning or crouching? Describe the posture.",
        "Report the body posture the person holds.",
        "Describe the stance and orientation of the person's body.",
        "What body posture do you observe? Answer in words.",
        "Give a detailed account of the person's posture.",
    ],
}

# Prefixes, keyed by which inputs a sample carries.
MOTION_PREFIX = DEFAULT_MOTION_TOKEN + "\n"
IMAGE_PREFIX = DEFAULT_IMAGE_TOKEN + "\n"
IMAGE_MOTION_PREFIX = DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_MOTION_TOKEN + "\n"

# What the dataset builder samples from. `mixed` mirrors the release's
# IMAGE_MOTION + IMAGE + MOTION concatenation; `motion` is the fallback for
# samples with no egocentric frame and for the stage-2 pretrain family.
CATEGORY_QUESTION_LISTS = {
    ct: {
        "mixed": ([IMAGE_MOTION_PREFIX + b for b in bodies]
                  + [IMAGE_PREFIX + b for b in bodies]
                  + [MOTION_PREFIX + b for b in bodies]),
        "motion": [MOTION_PREFIX + b for b in bodies],
    }
    for ct, bodies in CATEGORY_QUESTION_BODIES.items()
}

# The question used at evaluation: body [0] of each category with the
# image+motion prefix. As in the release (test_ego4o_image_imu_batch.py:245
# hardcodes IMAGE_MOTION_TO_TEXT_QUESTION_LIST[0]) this is one of the training
# questions, not a held-out one. The three body-part entries mirror MotionGPT3's
# EVAL_TEMPLATES_ME2T wording so both models are asked for the same kind of
# caption; atomic keeps the release's own question.
EVAL_QUESTION_BY_TYPE = {
    ct: IMAGE_MOTION_PREFIX + bodies[0]
    for ct, bodies in CATEGORY_QUESTION_BODIES.items()
}


