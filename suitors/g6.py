from typing import Dict

from flowers import Bouquet, Flower, FlowerSizes, FlowerColors, FlowerTypes
from suitors.base import BaseSuitor

import numpy as np
from constants import MAX_BOUQUET_SIZE
from utils import flatten_counter
from collections import Counter


class Suitor(BaseSuitor):
    def __init__(self, days: int, num_suitors: int, suitor_id: int):
        """
        :param days: number of days of courtship
        :param num_suitors: number of suitors, including yourself
        :param suitor_id: unique id of your suitor in range(num_suitors)
        """
        self.test_lengths = [[-1, MAX_BOUQUET_SIZE - 2]] * (num_suitors - 1) # tuple has (min, max) values for length feature
        # self.min_lengths = [-1] * (num_suitors - 1)
        # self.max_lengths = [MAX_BOUQUET_SIZE] * (num_suitors - 1)

        self.test_sizes = [0] * (num_suitors - 1) # 0 for Small, 1 for Medium, 2 for Large
        self.min_sizes = [[-1, -1, -1]] * (num_suitors - 1) # tuple for sizes (Small, Medium, Large)
        self.max_sizes = [[MAX_BOUQUET_SIZE, MAX_BOUQUET_SIZE, MAX_BOUQUET_SIZE]] * (num_suitors - 1)

        self.chosen_bouquets = [None] * (num_suitors - 1)
        self.last_bouquets = [None] * (num_suitors - 1)

        self.test_features = [0] * (num_suitors - 1) # 0 is length of bouquet, 1 is size of each flower, 2 is type of each flower, 3 is amount of each flower
        self.chosen_features = [[None, [None, None, None], None, None]] * (num_suitors - 1)

        super().__init__(days, num_suitors, suitor_id, name='g6')


    def _prepare_bouquet_lengths(self, remaining_flowers, recipient_id):
        # here, have already gotten best bouquet possible
        if self.chosen_bouquets[recipient_id] is not None:
            return self.chosen_bouquets[recipient_id]

        # using incremental method to find ideal length of all flowers used in bouquet
        length = self.test_lengths[recipient_id][0] + 2 # using minimum and working way up to larger bouquets
        num_remaining_flowers = sum([num for num in remaining_flowers.values()])

        # rest of choices are random
        if 0 < length < num_remaining_flowers:
            chosen_flowers = np.random.choice(flatten_counter(remaining_flowers), size=(length,), replace=False)
            chosen_flower_counts = dict(Counter(chosen_flowers))
            for k, v in chosen_flower_counts.items():
                remaining_flowers[k] -= v
                assert remaining_flowers[k] >= 0
        else:
            chosen_flower_counts = dict()
        chosen_bouquet = Bouquet(chosen_flower_counts)
        return self.suitor_id, recipient_id, chosen_bouquet

    def _prepare_bouquet_sizes(self, remaining_flowers, recipient_id):
        # here, have already gotten best bouquet possible
        if self.chosen_bouquets[recipient_id] is not None:
            return self.chosen_bouquets[recipient_id]

        length = int(self.chosen_features[recipient_id][0]) # best predicted length of bouquet
        print("recipient_id_bouquet_sizes: {}".format(recipient_id))
        if length > 0:
            # separating flowers by their size options
            size_split_flowers = [[], [], []]
            for flower in remaining_flowers:
                if remaining_flowers[flower] > 0:
                    if flower.size == FlowerSizes.Small:
                        size_split_flowers[0].append(flower)
                    elif flower.size == FlowerSizes.Medium:
                        size_split_flowers[1].append(flower)
                    elif flower.size == FlowerSizes.Large:
                        size_split_flowers[2].append(flower)

            # choosing how many of each size will be used
            # (e.g. if testing FlowerSize.Small, will use all FlowerSize.Small available, then use equally FlowerSize.Medium and FlowerSize.Large
            #   but if first time testing FlowerSize.Small, will start by testing 0 FlowerSize.small)
            chosen_flowers = list()
            possible_sizes = [0, 1, 2]
            min_size = self.min_sizes[recipient_id][self.test_sizes[recipient_id]]
            max_size = min([self.max_sizes[recipient_id][self.test_sizes[recipient_id]], len(size_split_flowers[self.test_sizes[recipient_id]])])
            flowers_added = 0
            size_add = 0

            # adding flowers if already decided on certain size
            print("self.chosen_features[{}][1]:".format(recipient_id))
            print(self.chosen_features[recipient_id][1])
            for chosen_size in self.chosen_features[recipient_id][1]:
                if chosen_size is None:
                    break
                flowers_of_wanted_size = size_split_flowers[possible_sizes[0]]
                for i in range(chosen_size):
                    chosen_flowers.append(flowers_of_wanted_size.pop())
                    flowers_added += 1
                del possible_sizes[0]

            print("possible sizes")
            print(possible_sizes)
            print("recipient_id: {}".format(recipient_id))
            print(self.test_sizes[recipient_id])
            # now possible_sizes only has the non-tested sizes
            possible_sizes.remove(self.test_sizes[recipient_id])

            print("flowers added: {}".format(flowers_added))
            print("chosen flowers")
            print(chosen_flowers)

            if min_size == -1: # first time testing this size, so want to start by testing 0
                print("part a")
                print("length: {}".format(length))
                self.min_sizes[recipient_id][self.test_sizes[recipient_id]] = 0 # indicating this is no longer the first testing of sizes
                while flowers_added < length:
                    print("size add: {}".format(size_add))
                    print("possible_sizes: {}".format(possible_sizes))
                    print("size_split_flowers")
                    print(size_split_flowers)
                    flowers_of_wanted_size = size_split_flowers[possible_sizes[size_add]]
                    if len(flowers_of_wanted_size) > 0:
                        chosen_flowers.append(flowers_of_wanted_size.pop()) # adding flower from list of that size to bouquet & removing it as option
                        flowers_added += 1
                    size_add = 0 if size_add == len(possible_sizes) - 1 else 1
                print("flowers added: {}".format(flowers_added))
            else: # not first time testing this size, so want to test, so want to test between size min and size max
                print("part b")
                size_length = int(min_size + (max_size - min_size) / 2)
                flowers_of_wanted_size = size_split_flowers[self.test_sizes[recipient_id]]
                while flowers_added < size_length and len(flowers_of_wanted_size) > 0: # adding requested flowers of size being tested (if enough flowers are available)
                    chosen_flowers.append(flowers_of_wanted_size.pop())
                    flowers_added += 1
                if flowers_added < size_length:
                    while flowers_added < length:
                        flowers_of_wanted_size = size_split_flowers[possible_sizes[size_add]]
                        if len(flowers_of_wanted_size) > 0: # adding flower from list of that size to bouquet & removing it as option
                            chosen_flowers.append(
                                flowers_of_wanted_size.pop())
                            flowers_added += 1
                        size_add = 0 if size_add == len(possible_sizes) - 1 else 1

            chosen_flower_counts = dict(Counter(chosen_flowers))
            print("chosen_flower_counts: ")
            print(chosen_flower_counts)
            print(chosen_flower_counts.items())
            print(remaining_flowers)
            for k, v in chosen_flower_counts.items():
                remaining_flowers[k] -= v
                assert remaining_flowers[k] >= 0
        else:
            chosen_flower_counts = dict()
        chosen_bouquet = Bouquet(chosen_flower_counts)
        return self.suitor_id, recipient_id, chosen_bouquet


    def prepare_bouquets(self, flower_counts: Dict[Flower, int]):
        """
        :param flower_counts: flowers and associated counts for available flowers
        :return: list of tuples of (self.suitor_id, recipient_id, chosen_bouquet)
        the list should be of length len(self.num_suitors) - 1 because you should give a bouquet to everyone
         but yourself

        To get the list of suitor ids not including yourself, use the following snippet:

        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        """
        all_ids = np.arange(self.num_suitors)
        recipient_ids = all_ids[all_ids != self.suitor_id]
        remaining_flowers = flower_counts.copy()
        num_remaining = sum(remaining_flowers.values())

        for i in range(len(recipient_ids)):
            # self.max_lengths[i] = min(MAX_BOUQUET_SIZE, num_remaining)
            if self.test_features[i] == 0:
                if self.test_lengths[i][0] == -1:
                    self.last_bouquets[i] = (self.suitor_id, recipient_ids[i], Bouquet(dict()))
                else:
                    self.last_bouquets[i] = self._prepare_bouquet_lengths(remaining_flowers, recipient_ids[i])
            elif self.test_features[i] == 1:
                self.last_bouquets[i] = self._prepare_bouquet_sizes(remaining_flowers, recipient_ids[i])

        return self.last_bouquets

    def zero_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 0
        """
        min_flower = Flower(
            size=FlowerSizes.Small,
            color=FlowerColors.White,
            type=FlowerTypes.Rose
        )
        return Bouquet({min_flower: 1})

    def one_score_bouquet(self):
        """
        :return: a Bouquet for which your scoring function will return 1
        """
        max_flower = Flower(
            size=FlowerSizes.Large,
            color=FlowerColors.Blue,
            type=FlowerTypes.Begonia
        )
        return Bouquet({max_flower: 1})

    def score_types(self, types: Dict[FlowerTypes, int]):
        """
        :param types: dictionary of flower types and their associated counts in the bouquet
        :return: A score representing preference of the flower types in the bouquet
        """
        if len(types) == 0:
            return 0.0

        avg_types = float(np.mean([x.value for x in flatten_counter(types)]))
        return avg_types / (3 * (len(FlowerTypes) - 1))

    def score_colors(self, colors: Dict[FlowerColors, int]):
        """
        :param colors: dictionary of flower colors and their associated counts in the bouquet
        :return: A score representing preference of the flower colors in the bouquet
        """
        if len(colors) == 0:
            return 0.0

        avg_colors = float(np.mean([x.value for x in flatten_counter(colors)]))
        return avg_colors / (3 * (len(FlowerColors) - 1))

    def score_sizes(self, sizes: Dict[FlowerSizes, int]):
        """
        :param sizes: dictionary of flower sizes and their associated counts in the bouquet
        :return: A score representing preference of the flower sizes in the bouquet
        """
        if len(sizes) == 0:
            return 0

        avg_sizes = float(np.mean([x.value for x in flatten_counter(sizes)]))
        return avg_sizes / (3 * (len(FlowerSizes) - 1))

    def receive_feedback(self, feedback):
        """
        :param feedback:
        :return: nothing
        """
        if self.test_features[0] == 0 and self.test_lengths[0][0] == -1:
            self.feedback.append(feedback)
            self.test_lengths = [[0, MAX_BOUQUET_SIZE - 2] for _ in self.test_lengths]
            return

        last_feedback = self.feedback[len(self.feedback) - 1]
        self.feedback.append(feedback)

        for i in range(len(feedback) - 1):
            if feedback[i][1] == 1:  # gave best bouquet possible
                self.chosen_bouquets[i] = self.last_bouquets[i]
                continue

            if self.test_features[i] == 0:  # looking at bouquet length feature
                min_feat = self.test_lengths[i][0]
                max_feat = self.test_lengths[i][1]
            elif self.test_features[i] == 1:  # looking at bouquet size feature
                min_feat = self.min_sizes[i][self.test_sizes[i]]
                max_feat = self.max_sizes[i][self.test_sizes[i]]

            if min_feat == max_feat - 1:  # means that we believe we found best value (no more values to test)
                if feedback[i][1] > last_feedback[i][1]:  # this score is best
                    best_feat = min_feat
                else:  # last score was best
                    best_feat = max_feat

                if self.test_features[i] == 0:
                    self.chosen_features[i][0] = best_feat
                    self.test_features[i] += 1
                    self.max_sizes[i] = [best_feat, best_feat, best_feat]
                elif self.test_features[i] == 1:
                    self.chosen_features[i][1][self.test_sizes[i]] = best_feat
                    if self.test_sizes[i] == 2:
                        self.test_features[i] += 1
                    print("now on self.test_sizes[{}]: {}".format(i, self.test_features[i]))
                    self.test_sizes[i] += 1
            else:  # means can still do better
                # score improved but still have more values to test
                if feedback[i][1] > last_feedback[i][1]:
                    if self.test_features[i] == 0:
                        if self.test_lengths[i][0] <= 8:
                            self.test_lengths[i][0] += 2
                        elif self.test_lengths[i][0] <= 9:
                            self.test_lengths[i][0] += 1
                    elif self.test_features[i] == 1:
                        self.min_sizes[i][self.test_sizes[i]] = int(min_feat + ((max_feat - min_feat) / 2))
                else:  # score got worse but still have more values to test
                    if self.test_features[i] == 0:
                        self.test_lengths[i][0] -= 1
                        self.test_lengths[i][1] = min_feat
                    elif self.test_features[i] == 1:
                        self.max_sizes[i][self.test_sizes[i]] = int(min_feat + ((max_feat - min_feat) / 2))


        # if self.min_lengths[0] == -1:
        #     self.min_lengths = [0 for _ in self.min_lengths] # first round of testing for this feature was done so now want to give it min of 0
        #     self.feedback.append(feedback)
        # else:
        #     last_feedback = self.feedback[len(self.feedback) - 1]
        #     self.feedback.append(feedback)
        #
        #     for i in range(len(feedback) - 1):
        #         if feedback[i][1] == 1: # gave best bouquet possible
        #             self.chosen_bouquets[i] = self.last_bouquets[i]
        #             continue
        #
        #         if self.test_features[i] == 0:  # looking at bouquet length feature
        #             min_feat = self.min_lengths[i]
        #             max_feat = self.max_lengths[i]
        #         elif self.test_features[i] == 1:  # looking at bouquet size feature
        #             min_feat = self.min_sizes[i][self.test_sizes[i]]
        #             max_feat = self.max_sizes[i][self.test_sizes[i]]
        #
        #         if min_feat == max_feat + 1: # means that we believe we found best value (no more values to test)
        #             if feedback[i][1] > last_feedback[i][1]:  # this score is best
        #                 best_feat = min_feat
        #             else: # last score was best
        #                 best_feat = max_feat
        #
        #             if self.test_features[i] == 0:
        #                 self.chosen_features[i][0] = best_feat
        #                 self.test_features[i] += 1
        #             elif self.test_features[i] == 1:
        #                 self.chosen_features[i][1][self.test_sizes[i]] = best_feat
        #                 if self.test_sizes[i] == 2:
        #                     self.test_features[i] += 1
        #         else: # means can still do better
        #             if feedback[i][1] > last_feedback[i][1]:  # score improved but still have more values to test
        #                 if self.test_features[i] == 0:
        #                     self.min_lengths[i] = int(min_feat + ((max_feat - min_feat) / 2))
        #                 elif self.test_features[i] == 1:
        #                     self.min_sizes[i][self.test_sizes[i]] = int(min_feat + ((max_feat - min_feat) / 2))
        #             else: # score got worse but still have more values to test
        #                 if self.test_features[i] == 0:
        #                     self.max_lengths[i] = int(min_feat + ((max_feat - min_feat) / 2))
        #                 elif self.test_features[i] == 1:
        #                     self.max_sizes[i][self.test_sizes[i]] = int(min_feat + ((max_feat - min_feat) / 2))


# if d > length(get_all_possible_bouquets()) can try every bouquet and always get #1
# if have best ranking but not a score of 1, may lose ranking trying to improve score -- may want to revert to best ranking on last day just to have better chance
# assumes linearity of how feature improves performance
# can easily run out of flowers -- need to make sure at least have enough flowers on last day
# if know you got someone's bouquet correctly, only need to give it to them on last day
