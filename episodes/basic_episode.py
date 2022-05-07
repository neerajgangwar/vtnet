""" Contains the Episodes for Navigation. """
import random
import torch

from data.constants import GOAL_SUCCESS_REWARD, STEP_PENALTY, DUPLICATE_STATE, UNSEEN_STATE
from data.constants import DONE
from data.environment import Environment
from data.constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_UP,
    LOOK_DOWN,
    DONE,
)

from .episode import Episode


class BasicEpisode(Episode):
    """ Episode for Navigation. """

    def __init__(self, args, device, strict_done=False):
        super(BasicEpisode, self).__init__()

        self._env = None
        self.device = device

        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None
        self.actions = [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_UP, LOOK_DOWN, DONE]
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self._last_action_embedding_idx = 0
        self.target_object = None
        self.prev_frame = None
        self.current_frame = None
        self.scene = None

        self.scene_states = []

        self._episode_times = 0
        self.seen_percentage = 0

        self.state_reps = []
        self.state_memory = []
        self.action_memory = []
        self.obs_reps = []

        self.episode_length = 0
        self.target_object_detected = False

        # tools
        self.states = []
        self.actions_record = []
        self.action_outputs = []
        self.detection_results = []

        # imitation learning
        # self.imitation_learning = args.imitation_learning
        self.action_failed_il = False

        self.action_probs = []

        # self.meta_learning = args.update_meta_network
        self.meta_predictions = []

        self.warm_up = False
        self.num_workers = args.num_workers
        self.episode_num = 0

    @property
    def environment(self):
        return self._env

    @property
    def actions_list(self):
        return [{"action": a} for a in self.actions]

    @property
    def episode_times(self):
        return self._episode_times

    @episode_times.setter
    def episode_times(self, times):
        self._episode_times = times

    def reset(self):
        self.done_count = 0
        self.duplicate_count = 0
        self._env.back_to_start()

    def state_for_agent(self):
        return self.environment.current_frame

    def current_detection_feature(self):
        return self.environment.current_detection_feature

    def current_agent_position(self):
        """ Get the current position of the agent in the scene. """
        return self.environment.current_agent_position

    def step(self, action_as_int):

        action = self.actions_list[action_as_int]

        if action["action"] != DONE:
            self.environment.step(action)
        else:
            self.done_count += 1

        reward, terminal, action_was_successful = self.judge(action)
        return reward, terminal, action_was_successful

    def judge(self, action):
        """ Judge the last event. """
        reward = STEP_PENALTY
        # Thresholding replaced with simple look up for efficiency.
        if self.environment.controller.state in self.scene_states:
            if action["action"] != DONE:
                if self.environment.last_action_success:
                    self.duplicate_count += 1
                else:
                    self.failed_action_count += 1
        else:
            self.scene_states.append(self.environment.controller.state)

        done = False

        if action["action"] == DONE:
            action_was_successful = False
            for id_ in self.task_data:
                if self.environment.object_is_visible(id_):
                    reward = GOAL_SUCCESS_REWARD
                    done = True
                    action_was_successful = True
                    break
        else:
            action_was_successful = self.environment.last_action_success

        return reward, done, action_was_successful

    # Set the target index.
    @property
    def target_object_index(self):
        """ Return the index which corresponds to the target object. """
        return self._target_object_index

    @target_object_index.setter
    def target_object_index(self, target_object_index):
        """ Set the target object by specifying the index. """
        self._target_object_index = torch.LongTensor([target_object_index]).to(self.device)

    def _new_episode(self, args, scenes, targets):
        """ New navigation episode. """
        scene = random.choice(scenes)
        self.scene = scene

        if self._env is None:
            self._env = Environment(
                offline_data_dir=args.data_dir,
                use_offline_controller=True,
                grid_size=0.25,
                detection_feature_file_name='{}_features_{}cls.hdf5'.format("detr", 22),
                images_file_name="resnet18_featuremap.hdf5",
                visible_object_map_file_name="visible_object_map_1.5.json",
                local_executable_path=None,
                optimal_action_file_name="optimal_action.json",
            )
            self._env.start(scene)
        else:
            self._env.reset(scene)

        self.task_data = []

        objects = self._env.all_objects()

        visible_objects = [obj.split("|")[0] for obj in objects]
        intersection = [obj for obj in visible_objects if obj in targets]

        idx = random.randint(0, len(intersection) - 1)
        goal_object_type = intersection[idx]
        self.target_object = goal_object_type

        for id_ in objects:
            type_ = id_.split("|")[0]
            if goal_object_type == type_:
                self.task_data.append(id_)

        # Randomize the start location.
        # self._env.randomize_agent_location()

        warm_up_path_len = 200
        if (self.episode_num * self.num_workers) < 500000:
            warm_up_path_len = 5 * (int((self.episode_num * self.num_workers) / 50000) + 1)
        else:
            self.warm_up = False

        if self.warm_up:
            for _ in range(10):
                self._env.randomize_agent_location()
                shortest_path_len = 1000
                for _id in self.task_data:
                    path_len = self._env.controller.shortest_path_to_target(self._env.start_state, _id)[1]
                    if path_len < shortest_path_len:
                        shortest_path_len = path_len
                if shortest_path_len <= warm_up_path_len:
                    break
        else:
            self._env.randomize_agent_location()

        if args.verbose:
            print("Scene", scene, "Navigating towards:", goal_object_type)

    def new_episode(self, args, scenes, targets):
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self.episode_length = 0
        self.prev_frame = None
        self.current_frame = None
        self.scene_states = []

        self.state_reps = []
        self.state_memory = []
        self.action_memory = []

        self.target_object_detected = False

        self.episode_times += 1
        self.episode_num += 1

        self.states = []
        self.actions_record = []
        self.action_outputs = []
        self.detection_results = []
        self.obs_reps = []

        self.action_failed_il = False

        self.action_probs = []
        self.meta_predictions = []

        self._new_episode(args, scenes, targets)
