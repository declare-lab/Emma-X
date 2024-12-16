from collections import defaultdict

import numpy as np
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import AutoTokenizer


class Solver:
    def __init__(self, action_tokenizer=None, verbose=True) -> None:
        self.verbose = verbose
        self.action_tokenizer = action_tokenizer
        self.coordinates_key = "NEXT GRIPPER:"
        self.movement_key = "MOVEMENT:"
        self.policy_key = "POLICIES:"

    def compare_movement(self, pred_pos, label_pos):

        dist = np.sum(np.abs(pred_pos - label_pos))
        relative_dist = np.sum(np.abs(dist / label_pos))
        return dist, relative_dist, dist == 0

    def compare_policy(self, pred_pol, label_pol):
        dist = 0
        cnt = 0
        for i in range(min(len(label_pol), len(pred_pol))):
            for j in range(len(label_pol[0])):
                dist += label_pol[i][j] == pred_pol[i][j]
                cnt += 1
        assert cnt % 7 == 0
        return dist / cnt

    def extract_2d_coordinates(self, text):
        try:
            coordinates_index = text.index(self.coordinates_key) + len(self.coordinates_key)
            coord = text[coordinates_index:]
            coord = [o for o in coord.split("\n") if len(o.strip()) != 0]
            coord = eval(coord[0].strip())
        except Exception:
            coord = [0, 0]
        return coord

    def extract_movement_plan(self, text):
        require_unorm = None
        try:
            # text after key word
            movement_index = text.index(self.movement_key) + len(self.movement_key)
            movement_level = text[movement_index:]
            movement_level = [o for o in movement_level.split("\n") if len(o.strip()) != 0]
            movement_level = movement_level[0].strip()

            if "gripper" not in movement_level:  # for normalized tokenized version
                require_unorm = True
                movement_token_ids = self.action_tokenizer.tokenizer(movement_level, add_special_tokens=False).input_ids
                movement_norm = self.action_tokenizer.decode_token_ids_to_actions(np.array(movement_token_ids))
                movement_norm = movement_norm[1:8]
                assert len(movement_norm) == 7
            else:  # for unnormalized text version
                require_unorm = False
                movement_level = [o for o in movement_level.split(";") if len(o) > 0]
                movement_level = movement_level[:7]

                position = defaultdict(int)
                movement_to_pos = dict(
                    move_backward=(-1, "y"),
                    move_forward=(1, "y"),
                    move_right=(-1, "x"),
                    move_left=(1, "x"),
                    move_downward=(-1, "z"),
                    move_upward=(1, "z"),
                    roll_downward=(-1, "ox"),
                    roll_upward=(1, "ox"),
                    swing_downward=(-1, "ox"),
                    swing_upward=(1, "ox"),
                    pitch_downward=(-1, "oy"),
                    pitch_upward=(1, "oy"),
                    yaw_downward=(-1, "oz"),
                    yaw_upward=(1, "oz"),
                    rotate_clockwise=(-1, "oz"),
                    rotate_counterclockwise=(1, "oz"),
                    close_gripper=(-1, "grip"),
                    open_gripper=(1, "grip"),
                )

                for ml in movement_level:
                    direction = "_".join(ml.split()[:2])
                    sign, axis = movement_to_pos[direction]
                    scale = 1
                    if "o" in axis:  # for orientation
                        scale = scale * 1e-3
                    elif "grip" in axis:  # for gripper
                        scale = scale
                    else:  # for xyz
                        scale = scale / 180 * np.pi

                    if "grip" in axis:
                        level = round("open" in ml)
                    else:
                        level = int(ml.split()[2])

                    position[axis] += sign * scale * level
                movement_norm = [position[idx] for idx in ["x", "y", "z", "ox", "oy", "oz", "grip"]]

        except:
            movement_norm = [-100] * 7

        return require_unorm, np.array(movement_norm)

    def extract_action_policies(self, text):
        try:
            if self.policy_key in text:

                policy_index = text.index(self.policy_key) + len(self.policy_key)
                policy = text[policy_index:]
                remain_text = text[: text.index(self.policy_key)]
                policies = [o for o in policy.split("\n") if len(o.strip()) != 0]
                policies = policies[0].strip()
            else:
                policies = text.strip()
                remain_text = ""

            policies_num = []
            for policy_text in policies.split(";"):
                policy_token = self.action_tokenizer.tokenizer(policy_text, add_special_tokens=False).input_ids
                action_policy = self.action_tokenizer.decode_token_ids_to_actions(np.array(policy_token))
                # The first token is meaningless
                action_policy = action_policy[1:]
                action_policy = action_policy[:7]
                # assert len(action_policy) == 7
                if len(action_policy) != 7:
                    action_policy = [0] * 7
                policies_num.append(action_policy.tolist())

        except:
            policies_num = [[0] * 7]
            remain_text = text

        return policies_num, remain_text

    def evaluate_single(self, ground_truth, prediction, verbose=False):
        gt_policies, ground_truth = self.extract_action_policies(ground_truth)
        pred_policies, prediction = self.extract_action_policies(prediction)

        _, pred_movement = self.extract_movement_plan(prediction)
        _, gt_movement = self.extract_movement_plan(ground_truth)

        dist, relative_dist, _ = self.compare_movement(label_pos=gt_movement, pred_pos=pred_movement)

        # pred_2d = self.extract_2d_coordinates(prediction)
        # gt_2d = self.extract_2d_coordinates(ground_truth)

        next_state_score = 0

        acc = self.compare_policy(label_pol=gt_policies, pred_pol=pred_policies)

        return next_state_score, acc, dist, relative_dist, pred_policies, gt_policies

    def evaluate_batch(self, batch_gt, batch_pred, verbose=False):
        state_acc_ls = []
        action_acc_ls = []
        L1_loss_ls = []
        relative_L1_loss_ls = []
        pred_policies_ls = []
        gt_policies_ls = []
        for i in range(len(batch_gt)):
            ground_truth = batch_gt[i]
            prediction = batch_pred[i]
            next_state_score, action_policy_score, L1_dist, relative_L1_dist, pred_policies, gt_policies = (
                self.evaluate_single(ground_truth, prediction)
            )
            state_acc_ls.append(next_state_score)
            action_acc_ls.append(action_policy_score)
            L1_loss_ls.append(L1_dist)
            relative_L1_loss_ls.append(relative_L1_dist)
            pred_policies_ls.append(pred_policies)
            gt_policies_ls.append(gt_policies)
            if verbose:
                print(f"Ground Truth:\n\n {ground_truth}")
                print()
                print(f"prediction:\n\n {prediction}")
                print()
                print(f"Ground Truth Policies:\n\n {gt_policies}")
                print(f"prediction policies:\n\n {pred_policies}")
                print("*" * 40)

        return state_acc_ls, action_acc_ls, L1_loss_ls, relative_L1_loss_ls, pred_policies_ls, gt_policies_ls


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", model_max_length=2048, padding_side="right")
action_tokenizer = ActionTokenizer(tokenizer)
solver = Solver(action_tokenizer)

if __name__ == "__main__":

    prediction = "REASONING:\nThe robot has grasped the pot and is raising it. Because needs to move the pot.\nSUBTASK: Lifting the pot\n\nNEXT GRIPPER: [105, 74]\n\nMOVEMENT:\nむ唐兴˜食嘉给\nPOLICIES:\n编给모构만效Ÿ;Ἐ명모식溪仮Ÿ\n".strip()
    ground_truth = "REASONING:\nThe robot has grasped the pot and is raising it. Because needs to move the pot.\nSUBTASK: Lifting the pot\n\nNEXT GRIPPER: [105, 74]\n\nMOVEMENT:\nむ唐兴˜食嘉给\nPOLICIES:\n编给모构만效Ÿ;Ἐ명모식溪仮Ÿ\n".strip()

    state_acc_ls, action_acc_ls, L1_loss_ls, relative_L1_loss_ls, pred_policies_ls, gt_policies_ls = (
        solver.evaluate_batch([ground_truth], [prediction])
    )
    print(pred_policies_ls)
    print(gt_policies_ls)
    print(action_acc_ls)

# (Pdb) prompt_str
# 'What action should the robot take to achieve the instruction\nINSTRUCTION: \nPut the pot next to the cans.\nCURRENT GRIPPER: [48, 63]\n'
# (Pdb) gpt_output
# 'REASONING:\nThe robot has grasped the pot and is raising it. Because needs to move the pot.\nSUBTASK: Lifting the pot\n\nNEXT GRIPPER: [105, 74]\n\nMOVEMENT:\nむ唐兴˜食嘉给\nPOLICIES:\n编给모构만效Ÿ;Ἐ명모식溪仮Ÿ\n'
