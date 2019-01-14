import numpy as np
import random
from collections import defaultdict
from longbinaryvectors import LongBinaryVector
from copy import deepcopy
import longbinaryvectors
import pickle


def conn(a, b):
    if a == b:
        return -1
    else:
        return 1


class Graph:
    vertexes = []

    def __init__(self, vertexes=None):
        if vertexes is None:
            vertexes = []

        self.vertexes = vertexes


class BinarySpace:

    def __init__(self, name, vector_length, gravity=True, gravity_alpha=1.0, change_threshold=0.0001, seed=0, end_count=100):
        self.name = name
        self.vector_length = vector_length
        self.gravity = gravity
        self.gravity_alpha = gravity_alpha
        self.change_threshold = change_threshold
        self.seed = seed
        self.end_count = end_count

        self.X = []
        self.graph_size = 0

    def minimize_energy_given_graph(self, graph):
        self.X = []
        X = self.X

        random.seed(self.seed)

        self.graph_size = len(graph.vertexes)

        for t in range(self.graph_size):
            x = LongBinaryVector(self.vector_length, self.seed)
            X.append(x)

        vertexes = graph.vertexes

        startX = deepcopy(X)

        distances = defaultdict(lambda: -1)
        vector_energy = np.zeros(self.graph_size)
        total_energy = self.compute_energy(X, vertexes, distances, vector_energy, first=True)
        total_start_energy = total_energy
        range_vector = list(range(self.vector_length))
        count = 1
        prev_count = 0
        num_changes = []
        look_back = self.graph_size
        history = []
        iterations = []
        energy = [total_start_energy]
        gravity_halving = False

        prev_energy_ratio = 1.0

        print('Starting Tension: ' + str(total_energy))

        iterations.append([total_energy, 1.0, []])

        end_count = self.end_count
        end_command = False

        while not end_command:
            if count >= end_count:
                user_input = input('Reached ' + str(end_count) +
                                   ' iterations, keep going? ("yes" or any other input for no.) --> ')

                if user_input == 'yes':
                    user_input = \
                        int(input('What should be the new max iterations count? (enter an int) --> '))

                    end_count = user_input
                else:
                    end_command = True

                    print('Ending program...')

            while count < end_count:
                # while sum(vector_energy) != 0:
                vector_probability = vector_energy / sum(vector_energy)
                row = np.random.choice(range(self.graph_size), 1, p=vector_probability)
                row = row[0]
                bit_gradient = np.ones(self.vector_length)
                m_row = vertexes[row][0]

                to_flip = 0
                changed = set()
                gradient_changes = -1

                altered = []
                
                safety_counter = 0
                while sum(bit_gradient) > 0:
                    safety_counter += 1
                    if (safety_counter > 10):
                        break

                    for bb in np.random.permutation(range_vector):
                        force_curr = 0
                        force_flip = 0

                        for vv in vertexes[row][1]:
                            if row != vv[0]:
                                m_vv = vertexes[vv[0]][0]

                                if self.gravity:
                                    if distances[(row, vv[0])] == 0:
                                        distance = longbinaryvectors.hamming_distance(X[row], X[vv[0]])
                                        distances[(row, vv[0])] = distance
                                    else:
                                        distance = distances[(row, vv[0])]

                                c_curr = conn(X[row].vector[0][bb], X[vv[0]].vector[0][bb])
                                c_flip = conn(not X[row].vector[0][bb], X[vv[0]].vector[0][bb])

                                if self.gravity:
                                    if X[row].vector[0][bb] == X[vv[0]].vector[0][bb]:
                                        distance_flip = distance + 1
                                    else:
                                        distance_flip = distance - 1

                                    if distance == 0:
                                        distance = 1

                                    if distance_flip == 0:
                                        distance_flip = 1

                                distance /= self.vector_length
                                if self.gravity:
                                    force_curr += distance * vv[1] * c_curr - self.gravity_alpha * m_row * m_vv * c_curr / \
                                                  (distance * distance)
                                    force_flip += distance * vv[1] * c_flip - self.gravity_alpha * m_row * m_vv * c_flip / \
                                                  (distance_flip * distance_flip)
                                else:
                                    force_curr += distance * vv[1] * c_curr
                                    force_flip += distance * vv[1] * c_flip

                        bit_gradient[bb] = max(max(force_curr, 0) - max(force_flip, 0), 0)

                        if bit_gradient[bb] / vector_energy[row] > self.change_threshold:
                            if self.gravity:
                                for vv in vertexes[row][1]:
                                    if X[row].vector[0][bb] == X[vv[0]].vector[0][bb]:
                                        distances[(row, vv[0])] += 1
                                    else:
                                        distances[(row, vv[0])] -= 1

                            X[row].vector[0][bb] = not X[row].vector[0][bb]

                            if bb in changed:
                                changed.remove(bb)
                            else:
                                changed.add(bb)
                        else:
                            bit_gradient[bb] = 0

                    print('     Row ' + str(row) + ', change: ' + str(sum(bit_gradient)))
                    #      ', bit positions changed: ' + str(changed))

                    altered.append([row, sum(bit_gradient), changed])

                    gradient_changes += 1

                num_changes.append(gradient_changes)
                samples = min(look_back, len(num_changes))
                average_changes = sum(num_changes[len(num_changes) - samples:]) / samples

                if average_changes > 0:
                    if average_changes < 0.5:
                        self.change_threshold /= 0.5 / average_changes
                    elif average_changes > 1.5:
                        self.change_threshold *= average_changes / 1.5
                else:
                    self.change_threshold /= 2

                if len(changed) > 0:
                    for vv in vertexes[row][2]:
                        for el in changed:
                            if X[row].vector[0][el] == X[vv].vector[0][el]:
                                distances[(row, vv)] -= 1
                            else:
                                distances[(row, vv)] += 1

                        total_energy -= vector_energy[vv]
                        total_energy = self.compute_energy_row(vv, total_energy, X, vertexes, distances, vector_energy)

                    total_energy -= vector_energy[row]
                    total_energy = self.compute_energy_row(row, total_energy, X, vertexes, distances, vector_energy)

                print('Count: ' + str(count) + ', Total start energy: ' + str(total_start_energy) +
                      ' , Current: ' + str(total_energy) + ', ' + str(total_energy / total_start_energy))

                iterations.append([total_energy, total_energy/total_start_energy, altered])

                history.append(total_energy / total_start_energy)
                energy.append(total_energy)

                if gravity_halving:
                    prev_count += 1

                    if prev_count > self.graph_size and \
                            (max(prev_energy_ratio - sum(history[count - self.graph_size:]) / self.graph_size, 0)) < 0.01:
                        self.gravity_alpha /= 2.0

                        print('HALVING GRAVITY CONSTANT!')

                        total_start_energy = self.compute_energy(X, vertexes, distances, vector_energy)
                        total_energy = total_start_energy

                        print('New total start energy: ' + str(total_energy))

                        prev_count = 0
                    elif prev_count >= self.graph_size:
                        prev_energy_ratio = history[count - self.graph_size]

                count += 1

        true_total_energy = self.compute_energy(X, vertexes, distances, vector_energy)

        print('True total energy: ' + str(true_total_energy))

        dump = [startX, X, iterations, true_total_energy]

        with open(self.name + '_minimized_vectors.pickle', 'wb') as handle:
            pickle.dump(dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def compute_energy_row(self, u, total, X, vertexes, distances, vector_energy):
        vector_total = 0

        for b in range(self.vector_length):
            force = 0
            mu = vertexes[u][0]

            for v in vertexes[u][1]:
                if u != v[0]:
                    mv = vertexes[v[0]][0]

                    if self.gravity:
                        if distances[(u, v[0])] == -1:
                            d = longbinaryvectors.hamming_distance(X[u], X[v[0]])
                            distances[(u, v[0])] = d
                        else:
                            d = distances[(u, v[0])]

                    c = conn(X[u].vector[0][b], X[v[0]].vector[0][b])

                    if self.gravity:
                        if d == 0:
                            d = 1

                    if self.gravity:
                        force -= self.gravity_alpha * mu * mv * c / (d * d)

                    force += mu * v[1] * conn(X[u].vector[0][b], X[v[0]].vector[0][b])

            total += max(force, 0)
            vector_total += max(force, 0)

        vector_energy[u] = vector_total

        return total

    def compute_energy(self, X, vertexes, distances, vector_energy, first=False):
        total = 0
        nodes = len([x for x in range(len(X) - 1)])

        for u in range(len(X) - 1):
            if first:
                print('Running node ' + str(u) + ' of ' + str(nodes) + '...')

            total = self.compute_energy_row(u, total, X, vertexes, distances, vector_energy)

        return total

    def create_graph_equal_spacing(self, size):
        node_set = [x for x in range(0, size)]
        vertexes = []
        #vertexes = [[size, [], node_set]]

        for node in node_set:
            connections = node_set[0:node] + node_set[node + 1:]

            for i in range(len(connections)):
                connections[i] = [connections[i], (len(node_set) - node) / (2.0 * size)]

            connections.append([size, 0.5])

            vertexes.append([1.0 / (2.0 * size), connections, []])

        vertexes.append([1 / 2.0, [], node_set])
        return Graph(vertexes)
