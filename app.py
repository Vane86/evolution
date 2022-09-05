import pygame
import numpy as np
import time


START_ENERGY = 30
START_DEAD_BODIES = 1
SPAWN_PROB_PER_DEAD = 1 / 10000
DAY_LIGHT_CYCLE_SPEED = 1 / 1000  # screens per tick

MUTATION_FACTOR = 0.5
BRAIN_HIDDEN_LAYER_SIZE = 10

MOVE_ENERGY_COST = 1
ROTATE_ENERGY_COST = 0.5
EAT_ENERGY_COST = 2
PHOTO_ENERGY_COST = 0.25
REPRODUCE_ENERGY_COST = 1

PHOTO_ENERGY_GAIN = 1
EAT_ENERGY_GAIN = 10

ENERGY_REPRODUCE_FACTOR = 0.2

MIN_ENERGY_BEFORE_DIE = 0
MAX_ENERGY_BEFORE_DIE = 500

GRID_SIZE = (50, 50)


class Simulation:

    def __init__(self, display_size, tick_period):
        pygame.init()

        self._screen = pygame.display.set_mode(display_size)
        self._display_size = display_size
        self._running = True
        self._time_since_update = 0
        self._tick_period = tick_period

        self._world = World(np.array(GRID_SIZE, dtype=int))
        self._selected_agent = None
        self._saved_brain = None

    def _process_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                if self._selected_agent is not None:
                    with open(f'genome_{int(time.time() * 1000)}.txt', mode='w') as out_file:
                        print('[' + ' '.join(map(str, self._selected_agent.get_genome())) + ']', file=out_file)
            elif event.key == pygame.K_c:
                if self._selected_agent is not None:
                    self._saved_brain = self._selected_agent.get_brain()
            elif event.key == pygame.K_1:
                self._world.toggle_draw_layer(0)
            elif event.key == pygame.K_2:
                self._world.toggle_draw_layer(1)
            elif event.key == pygame.K_3:
                self._world.toggle_draw_layer(2)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_grid_pos = (event.pos[1] // (self._display_size[1] // GRID_SIZE[0]),
                              event.pos[0] // (self._display_size[0] // GRID_SIZE[1]))
            if event.button == 1:
                self._world.add_agent(generate_random_agent(*mouse_grid_pos, brain=self._saved_brain))
            elif event.button == 3:
                self._selected_agent = self._world.get_agent(mouse_grid_pos)

    def _loop(self, dt):
        self._time_since_update += dt
        if self._time_since_update >= self._tick_period:
            self._time_since_update = 0
            self._screen.fill('black')
            self._world.tick(self._screen)
            if self._selected_agent is not None:
                print('Selected agent\'s energy:', self._selected_agent.get_energy())
                print('Selected agent\'s last action:', self._selected_agent.get_readable_last_action())
                if self._selected_agent.is_dead():
                    self._selected_agent = None
                    print('Selected agent is dead!')

    def start(self):
        clock = pygame.time.Clock()
        while self._running:
            for event in pygame.event.get():
                self._process_event(event)
            dt = clock.tick()

            self._loop(dt)
            pygame.display.flip()

        pygame.quit()


class World:

    def __init__(self, size):
        self._agent_grid = [[None] * size[1] for _ in range(size[0])]
        self._light_intensity_grid = [[1] * size[1] for _ in range(size[0])]
        for i in range(size[0] // 2, size[0]):
            self._light_intensity_grid[i] = [0] * size[1]
        self._dead_bodies_grid = [[START_DEAD_BODIES] * size[1] for _ in range(size[0])]

        self._draw_layers = [True, True, True]

        self._size = size
        self._age_ticks = 0

    def copy(self):
        cp = World(self._size)
        cp._agent_grid = [[self._agent_grid[i][j] if self._agent_grid[i][j] is not None else None for j in range(self._size[1])] for i in range(self._size[0])]
        cp._light_intensity_grid = [[self._light_intensity_grid[i][j] for j in range(self._size[1])] for i in range(self._size[0])]

        return cp

    def _floor_vec(self, vec):
        return np.mod(np.array(np.floor(vec), dtype=int), self._size)

    def is_in_grid(self, position):
        return 0 <= position[0] < self._size[0] and 0 <= position[1] < self._size[1]

    def get_agent(self, position):
        position = self._floor_vec(position)
        if 0 <= position[0] < self._size[0] and 0 <= position[1] < self._size[1]:
            return self._agent_grid[position[0]][position[1]]
        return None

    def get_light_intensity(self, position):
        position = self._floor_vec(position)
        return self._light_intensity_grid[position[0]][position[1]]

    def move_agent(self, old_pos, new_pos):
        old_pos = self._floor_vec(old_pos)
        new_pos = self._floor_vec(new_pos)
        if self._agent_grid[new_pos[0]][new_pos[1]] is not None:
            return None
        self._agent_grid[new_pos[0]][new_pos[1]] = self._agent_grid[old_pos[0]][old_pos[1]]
        self._agent_grid[old_pos[0]][old_pos[1]] = None
        return new_pos

    def add_agent(self, agent):
        ap = self._floor_vec(agent.get_position())
        if self._agent_grid[ap[0]][ap[1]] is not None:
            return None
        self._agent_grid[ap[0]][ap[1]] = agent
        return ap

    def fill_free_space(self, prob):
        for i in range(self._size[0]):
            for j in range(self._size[1]):
                if self._agent_grid[i][j] is None:
                    if np.random.random() < prob:
                        self._agent_grid[i][j] = generate_random_agent(i, j)

    def toggle_draw_layer(self, index):
        self._draw_layers[index] = not self._draw_layers[index]

    def tick(self, surface):
        world_copy = self.copy()
        for i in range(world_copy._size[0]):
            for j in range(world_copy._size[1]):
                if world_copy._agent_grid[i][j] is not None:
                    world_copy._agent_grid[i][j].update(world_copy, self)
        for i in range(self._size[0]):
            for j in range(self._size[1]):
                if self._draw_layers[0]:
                    pygame.draw.rect(surface, (255 * self._light_intensity_grid[i][j],) * 3,
                                     (10 * j, 10 * i, 10, 10))
                if self._draw_layers[2]:
                    pygame.draw.rect(surface, (min(self._dead_bodies_grid[i][j], 10) / 10 * 255, 0, 0),
                                     (10 * j, 10 * i, 10, 10))
                if self._agent_grid[i][j] is not None:
                    if self._draw_layers[1]:
                        self._agent_grid[i][j].draw(surface, 10)
                    if self._agent_grid[i][j].is_dead():
                        self._agent_grid[i][j] = None
                        self._dead_bodies_grid[i][j] += 1

        self._age_ticks += 1

        # if self._age_ticks % 500 == 0:
        #     self.fill_free_space(0.25)

        for i in range(self._size[0]):
            for j in range(self._size[1]):
                prob = self._dead_bodies_grid[i][j] * SPAWN_PROB_PER_DEAD
                if np.random.random() < prob and self._agent_grid[i][j] is None:
                    self._agent_grid[i][j] = generate_random_agent(i, j)
                    self._dead_bodies_grid[i][j] = 0  # max(0, self._dead_bodies_grid[i][j] - 5)

        self._light_intensity_grid = [[(1 + np.sin(i / self._size[0] * 6.28 + self._age_ticks * DAY_LIGHT_CYCLE_SPEED * 2 * 3.14)) / 2 for j in range(self._size[1])] for i in range(self._size[0])]


# Действия:
# шаг
# поворот по часовой
# поворот против часовой
# съесть
# фотосинтез
# размножиться
#
# Входы:
# кол-во энергии
# есть ли кто-то спереди
# есть ли кто-то сзади
# есть ли кто-то слева
# есть ли кто-то справа
# степень освещенности клетки
# 6 фиктивных переменных предыдущего действия
# успешность предыдущего действия
class Agent:

    def __init__(self, start_row, start_col, start_energy, start_color, mutation_factor=MUTATION_FACTOR, brain=None):
        self._position = np.array([start_row, start_col], dtype=float)
        self._energy = start_energy
        self._color = start_color

        act = lambda x: np.tanh(x)
        # act = lambda x: 1 / (1 + np.exp(-x))
        # act = lambda x: np.array(2 * (x > 0) - 1, dtype=float)
        # act = lambda x: 2 * x * (x > 0) + 0.5 * x * (x <= 0)

        self._brain = brain or AgentBrain([AgentBrainLayer(16, BRAIN_HIDDEN_LAYER_SIZE, act, mutation_factor),
                                           AgentBrainLayer(BRAIN_HIDDEN_LAYER_SIZE, BRAIN_HIDDEN_LAYER_SIZE, act, mutation_factor),
                                           AgentBrainLayer(BRAIN_HIDDEN_LAYER_SIZE, 6, act, mutation_factor)])

        self._direction = np.random.randint(0, 4)  # 0 - up, 1 - right, 2 - down, 3 - left
        self._is_dead = False
        self._last_action = 0
        self._last_action_success = True

        self._life_time = 0

    # d = 1 => cw, d = -1 => ccw
    def _rotate(self, d):
        self._direction = (self._direction + d) % 4
        self._energy -= ROTATE_ENERGY_COST
        return True

    def _get_front_position(self):
        result = np.array(self._position)
        if self._direction == 0:
            result[0] -= 1
        elif self._direction == 1:
            result[1] += 1
        elif self._direction == 2:
            result[0] += 1
        elif self._direction == 3:
            result[1] -= 1
        return result

    def _get_back_position(self):
        result = np.array(self._position)
        if self._direction == 0:
            result[0] += 1
        elif self._direction == 1:
            result[1] -= 1
        elif self._direction == 2:
            result[0] -= 1
        elif self._direction == 3:
            result[1] += 1
        return result

    def _get_right_position(self):
        result = np.array(self._position)
        if self._direction == 0:
            result[1] += 1
        elif self._direction == 1:
            result[0] += 1
        elif self._direction == 2:
            result[1] -= 1
        elif self._direction == 3:
            result[0] -= 1
        return result

    def _get_left_position(self):
        result = np.array(self._position)
        if self._direction == 0:
            result[1] -= 1
        elif self._direction == 1:
            result[0] -= 1
        elif self._direction == 2:
            result[1] += 1
        elif self._direction == 3:
            result[0] += 1
        return result

    def _move(self, world):
        new_position = self._get_front_position()
        if (result := world.move_agent(self._position, new_position)) is not None:
            self._position = result
            self._energy -= MOVE_ENERGY_COST
            return True
        self._energy -= MOVE_ENERGY_COST
        return False

    def _photo(self, light_intensity):
        self._energy += light_intensity * PHOTO_ENERGY_GAIN - PHOTO_ENERGY_COST
        return light_intensity * PHOTO_ENERGY_GAIN - PHOTO_ENERGY_COST > 0

    def _eat(self, agent):
        self._energy -= EAT_ENERGY_COST
        if agent is not None:
            prob = 1  # 1 / (1 + np.exp((-self._energy - 10 + agent.get_energy()) / 10))
            if np.random.random() <= prob:
                self._energy += agent.get_energy() + EAT_ENERGY_GAIN
                agent.kill()
                return True
        return False

    def _reproduce(self):
        self._energy -= REPRODUCE_ENERGY_COST
        new_color = (self._color + np.random.randint(-1, 1, 3)) * np.random.uniform(0.9, 1.1, 3)
        new_color = np.array([min(255, max(new_color[i], 0)) for i in range(3)])
        new_agent = Agent(*self._get_back_position(), self._energy * ENERGY_REPRODUCE_FACTOR, new_color, brain=self._brain.replicate())
        return new_agent

    def update(self, world_to_read, world_to_write):
        # inputs = (self._energy / 10,
        #           world_to_read.get_agent(self._get_front_position()) is not None,
        #           world_to_read.get_agent(self._get_back_position()) is not None,
        #           world_to_read.get_agent(self._get_right_position()) is not None,
        #           world_to_read.get_agent(self._get_left_position()) is not None,
        #           world_to_read.get_light_intensity(self._position),
        #           *(np.arange(0, 6, 1) == self._last_action),
        #           self._last_action_success)
        energy_features = (self._energy < 10, 10 <= self._energy < 100, 100 <= self._energy < 450, self._energy >= 450)
        inputs = (*energy_features,
                  world_to_read.get_agent(self._get_front_position()) is not None,
                  world_to_read.get_agent(self._get_back_position()) is not None,
                  world_to_read.get_agent(self._get_right_position()) is not None,
                  world_to_read.get_agent(self._get_left_position()) is not None,
                  world_to_read.get_light_intensity(self._position),
                  *(np.arange(0, 6, 1) == self._last_action),
                  self._last_action_success)
        action = self._brain.get_action(*inputs)
        # print(action)
        # action = np.random.randint(0, 6)
        # print('Energy:', self._energy)
        # print('Action:', action)
        if action == 0:
            self._last_action_success = self._move(world_to_write)
        elif action == 1:
            self._last_action_success = self._rotate(1)
        elif action == 2:
            self._last_action_success = self._rotate(-1)
        elif action == 3:
            eaten_agent = world_to_read.get_agent(self._get_front_position())
            self._last_action_success = self._eat(eaten_agent)
        elif action == 4:
            self._last_action_success = self._photo(world_to_read.get_light_intensity(self._position))
        elif action == 5:
            new_agent = self._reproduce()
            if self._energy >= 10:
                if world_to_write.add_agent(new_agent) is not None:
                    self._energy *= (1 - ENERGY_REPRODUCE_FACTOR)
                    self._last_action_success = True
                    # self._rotate(np.random.choice([-1, 1]))
                    pass
                else:
                    self._last_action_success = False
            else:
                self._last_action_success = False
        self._last_action = action

        if self._energy < MIN_ENERGY_BEFORE_DIE or self._energy > MAX_ENERGY_BEFORE_DIE:
            self.kill()

        self._life_time += 1

    def draw(self, surface, side_length_pixels):
        pygame.draw.rect(surface, tuple(self._color), (np.floor(self._position[1] * side_length_pixels),
                                                       np.floor(self._position[0] * side_length_pixels),
                                                       side_length_pixels,
                                                       side_length_pixels))
        fp = (self._get_front_position() - self._position) / 2 + self._position
        pygame.draw.line(surface, 255 - self._color,
                         (np.floor((self._position[1] + 0.5) * side_length_pixels),
                          np.floor((self._position[0] + 0.5) * side_length_pixels)),
                         (np.floor((fp[1] + 0.5) * side_length_pixels),
                          np.floor((fp[0] + 0.5) * side_length_pixels)))

    def get_energy(self):
        return self._energy

    def get_position(self):
        return self._position

    def get_genome(self):
        return self._brain.get_genome()

    def get_brain(self):
        return self._brain

    def get_readable_last_action(self):
        return ['MOVE', 'ROT_CW', 'ROT_CCW', 'EAT', 'PHOTO', 'REPR'][self._last_action]

    def kill(self):
        self._is_dead = True

    def is_dead(self):
        return self._is_dead


class AgentBrainLayer:

    def __init__(self, n_ins, n_outs, act, mutation_factor, *, mat=None, bias=None):
        self._mat = mat if mat is not None else np.random.uniform(-1, 1, (n_outs, n_ins))
        self._bias = bias if bias is not None else np.random.uniform(-1, 1, (n_outs, 1))
        self._act = act
        self._mf = mutation_factor

    def forward(self, x):
        x = self._act(self._mat @ x + self._bias)
        return x

    def replicate(self):
        new_mat = self._mat.copy()
        new_bias = self._bias.copy()
        mat_mut_mask = np.random.random(size=new_mat.shape) <= self._mf
        # new_mat[mat_mut_mask] = (2 * (np.random.random() > self._mf) - 1) * new_mat[mat_mut_mask]
        # new_mat[mat_mut_mask] = np.random.uniform(0, 2) * new_mat[mat_mut_mask]

        # new_mat[mat_mut_mask] += (2 * (np.random.random(size=new_mat[mat_mut_mask].shape) > self._mf) - 1) * self._mf * new_mat[mat_mut_mask]
        # new_mat[mat_mut_mask] = np.random.uniform(1 - self._mf, 1 + self._mf) * new_mat[mat_mut_mask]

        new_mat[mat_mut_mask] += np.random.uniform(-self._mf, self._mf, size=new_mat[mat_mut_mask].shape)

        bias_mut_mask = np.random.random(size=new_bias.shape) <= self._mf
        # new_bias[bias_mut_mask] = (2 * (np.random.random() > self._mf) - 1) * new_bias[bias_mut_mask]
        # new_bias[bias_mut_mask] = np.random.uniform(0, 2) * new_bias[bias_mut_mask]

        new_bias[bias_mut_mask] += np.random.uniform(-self._mf, self._mf, size=new_bias[bias_mut_mask].shape)

        return AgentBrainLayer(0, 0, self._act, self._mf, mat=new_mat, bias=new_bias)

    def get_genome(self):
        return np.concatenate((self._mat.reshape(-1), self._bias.reshape(-1)), axis=None)


class AgentBrain:

    def __init__(self, layers):
        self._layers = layers

    def _forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def get_output(self, *inputs):
        return self._forward(np.array(inputs).reshape(-1, 1)).reshape(-1)

    def get_action(self, *inputs):
        # print(self._forward(np.array(inputs).reshape(-1, 1)))
        return np.argmax(self._forward(np.array(inputs).reshape(-1, 1)))

    def replicate(self):
        new_layers = [layer.replicate() for layer in self._layers]
        return AgentBrain(new_layers)

    def get_genome(self):
        return np.concatenate([layer.get_genome() for layer in self._layers], axis=None)


def generate_random_agent(row, col, brain=None):
    color = pygame.Color(0, 0, 0, 0)
    color.hsva = (np.random.randint(0, 360), np.random.randint(0, 100), 100, 100)
    return Agent(row, col, np.random.uniform(0.5 * START_ENERGY, 1.5 * START_ENERGY), np.array([color.r, color.g, color.b], dtype=int), brain=brain)

