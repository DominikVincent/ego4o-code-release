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
