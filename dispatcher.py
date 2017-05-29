# -*- coding: utf-8 -*
from sklearn.preprocessing import scale
from sklearn.datasets import make_moons
import os
import glob
from micro_module import MicroModule

class DataGenerator:
    def __init__(self):
        pass

    def get_next_point(self):
        X, Y = make_moons(noise=0.2, random_state=0, n_samples=1)
        return X, Y

    def get_n_points(self, n_samples=5):
        X, Y = make_moons(noise=0.2, random_state=0, n_samples=n_samples)
        return X, Y


class Dispatcher:
    def __init__(self, num_modules):
        self.data_generator = DataGenerator()
        self.modules = {}
        for i in range(num_modules):
            self.modules[i] = MicroModule(module_id=i)

    def simple_initialisation(self):
        # насемплим немного образцов
        # раскидаем их случайно (?) по модулям
        # обучим модули
        for id, module in self.modules.items():
            X, Y = self.data_generator.get_n_points()
            module.set_episodic_memory(X, Y)
            module.learn(1400)
            module.episodic_memory.clean()

    def feed_next_data_point_to_modules(self):
        """
        1. генерирует точку данных
        2. скармливает ее всем модулям и сморит их уверенности
        3. тому модулю, который отвечает адекватней всех, эта точка добавляется в память
        """
        X, Y = self.data_generator.get_next_point()
        real_ans = Y[0]
        right_answers_unsertainties = {}
        wrong_answers_unsertainties = {}
        for key, module in self.modules.items():
            smooth_ans, sertainty = module.feed_episode(X)
            strict_ans = (smooth_ans>0.5)
            if strict_ans == real_ans:
                right_answers_unsertainties[module.module_id] = sertainty
            else:
                wrong_answers_unsertainties[module.module_id] = sertainty
        assert len(wrong_answers_unsertainties) + len(right_answers_unsertainties) == len(self.modules)
        # если есть правильно ответившие модули, то берем тот, у котрого высшая уверенность,
        # а если все ответили неправильно, то берем с низшей уверенностью
        # TODO можно реализовать распределение вероятностей по модулям пропорционально месту в посортированном массиве
        winner_id = -1
        if len(right_answers_unsertainties) > 0:
            sorted_ids = sorted(right_answers_unsertainties.items(),
                                key=lambda x: x[1])  # сначала маленькие, потом ебольшие
            winner_id = sorted_ids[0][0]
        else:
            sorted_ids = sorted(wrong_answers_unsertainties.items(), key=lambda x: x[1],
                                reverse=True)  # сначала большие, потом маленькие
            winner_id = sorted_ids[0][0]
        assert winner_id != -1
        self.modules[winner_id].add_episode_to_memory(X, Y)

    def try_consolidation(self):
        for key, module in self.modules.items():
            module.try_consolidation()

    def setup_folder_for_results(self, main_folder='results' ):
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)
        else:
            files = glob.glob('/' + main_folder + '/*')
            for f in files:
                os.remove(f)
        os.chdir(main_folder)