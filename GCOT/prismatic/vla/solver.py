from collections import defaultdict

import numpy as np
from prismatic.models.materialize import get_llm_backbone_and_tokenizer
from prismatic.vla.action_tokenizer import ActionTokenizer


import json


class Solver:
    def __init__(self, action_tokenizer, verbose=True) -> None:
        self.verbose = verbose
        self.action_tokenizer = action_tokenizer

    def compare_position(self, pred_pos, label_pos):

        dist, relative_dist = 0, 0
        for axis, scale in label_pos.items():
            if scale == 0:
                if pred_pos[axis] != 0:
                    relative_dist += abs((scale - pred_pos[axis]) / pred_pos[axis])
            else:
                relative_dist += abs((scale - pred_pos[axis]) / scale)

            dist += abs((scale - pred_pos[axis]))

        return dist, relative_dist, dist == 0

    def compare_policy(self, pred_pol, label_pol):
        dist = 0
        for i in range(min(len(label_pol), len(pred_pol))):
            for j in range(len(label_pol[0])):
                dist += abs(label_pol[i][j] - pred_pol[i][j])
        return dist / len(label_pol)

    def extract_2d_coordinates(self, text):
        coordinates_key = "Next position of the gripper in the image:"
        try:
            coordinates_index = text.index(coordinates_key) + len(coordinates_key)
            coord = text[coordinates_index:]
            coord = [o for o in coord.split("\n") if len(o.strip()) != 0]
            coord = eval(coord[0].strip())
        except Exception:
            coord = [0, 0]
        return coord

    def extract_movement_plan(self, text):
        position = defaultdict(int)
        movement_key = "Movement for next action:"
        try:
            # text after key word
            movement_index = text.index(movement_key) + len(movement_key)
            movement_level = text[movement_index:]
            movement_level = [o for o in movement_level.split("\n") if len(o.strip()) != 0]
            movement_level = movement_level[0].strip()
            movement_level = [o for o in movement_level.split(";") if len(o) > 0]
            movement_level = movement_level[:7]

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
                scale = int(ml.split()[2])
                position[axis] += sign * scale
        except Exception as e:
            pass

        return position

    def extract_action_policies(self, text):
        try:
            policy_key = "Next action policies:"
            policy_index = text.index(policy_key) + len(policy_key)
            policy = text[policy_index:]
            remain_text = text[: text.index(policy_key)]
            policies = [o for o in policy.split("\n") if len(o.strip()) != 0]
            policies = policies[0].strip()

            policies_num = []
            for policy_text in policies.split(";"):
                policy_token = self.action_tokenizer.tokenizer(policy_text, add_special_tokens=False).input_ids
                action_policy = self.action_tokenizer.decode_token_ids_to_actions(np.array(policy_token))
                # TODO small bug here:
                action_policy = action_policy[1:]
                assert len(action_policy) == 7
                policies_num.append(action_policy)

        except:
            policies_num = [[0] * 7]
            remain_text = text

        return policies_num, remain_text

    def evaluate_single(self, ground_truth, prediction, verbose=False):
        gt_policies, ground_truth = self.extract_action_policies(ground_truth)
        pred_policies, prediction = self.extract_action_policies(prediction)

        pred_position = self.extract_movement_plan(prediction)
        gt_position = self.extract_movement_plan(ground_truth)

        # pred_2d = self.extract_2d_coordinates(prediction)
        # gt_2d = self.extract_2d_coordinates(ground_truth)

        next_state_score = 0

        dist, relative_dist, _ = self.compare_position(label_pos=gt_position, pred_pos=pred_position)
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


if __name__ == "__main__":
    prediction = "Reasoning:\nThe robot needs to move its arm to the drawer handle in order to grab it\nNext action: Move to drawer handle\n\nNext position of the gripper in the image: [85,   113]\n\nMovement for next action:\nmove forward 0 steps; move left 0 steps; move downward 0 steps; roll downward 0 steps; pitch upward 0 steps; yaw counterclockwise 0 steps; close gripper 0     steps; \n\nNext action policies:\n\u0178\u0178\u0178\u0178\u0178\u0178\u5fe0".strip()
    ground_truth = "Reasoning:\nthe robot arm needs to move closer to the drawer to reach the handle\nNext action: approaching drawer\n\nNext position of the gripper in the image: [91,156]\n\nMovement for next action:\nmove forward 3 steps; move right 0 steps; move downward 10 steps; roll downward 15 steps; pitch downward 27 steps; yaw counterclockwise 90 steps; close gripper 0 steps; \n\nNext action policies:\n\u5b5d\u0388\u0178\u02da\u5b5d\u09cb\u5fe0;\u1ef1\u1ef1\u0388\u2593\u5b5d\u5065\u5fe0;\u0388\u0388\u0388\u1ef1\u0178\u98df\u5fe0;\u0178\u0388\u1ef1\u98db\u1ef1\ud130\u5fe0;\u8ad6\u0178\u1ef1\u0178\u5b5d\u0cb0\u5fe0;\u8ad6\u8ad6\u0388\u5b5d\u98db\u9648\u5fe0;\u8ad6\u8ad6\u1ef1\u1ef1\u02da\u5065\u5fe0;\u8ad6\u8ad6\u0388\u0388\u1ef1\u519b\u5fe0;\u8ad6\u8ad6\u0388\u0388\u5b5d\u5065\u5fe0;\u8ad6\u8ad6\u0388\u8ad6\u0388\u519b\u5fe0;\u8ad6\u0178\u0388\u0178\u0388\u519b\u5fe0;\u0126\u0178\u0178\u0388\u1ef1\u7ea2\u5fe0;\u8ad6\u0178\u0178\u0178\u0388\u0126\u5fe0;\u8ad6\u8ad6\u0178\u8ad6\u0388\u0126\u5fe0\n".strip()
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        "llama2-7b-pure",
    )
    action_tokenizer = ActionTokenizer(tokenizer)
    solver = Solver(action_tokenizer)
    path = "/mnt/data1/emrys/openvla/Logs/prism-dinosiglip-224px+mx-pred-all-2gpu+n0+b8+x7/validation_results/epoch_8_0.json"
    with open(path) as f:
        output = json.load(f)

    state_acc_ls, action_acc_ls, L1_loss_ls, relative_L1_loss_ls, pred_policies_ls, gt_policies_ls = solver.evaluate_batch([o["Ground Truth"] for o in output], [o["Prediction"] for o in output])
    print(action_acc_ls)
    print(np.mean(action_acc_ls))
