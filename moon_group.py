# -*- coding: utf-8 -*
import model_visualiser
import dispatcher
data_gen = dispatcher.DataGenerator()
realX, realY = data_gen.get_n_points(10)

def experiment_1():
    my_dispatcher = dispatcher.Dispatcher(num_modules=5)
    my_dispatcher.simple_initialisation()

    for i in range(100):
        my_dispatcher.feed_next_data_point_to_modules()
        my_dispatcher.try_consolidation()
        if i%10 == 0:
            for key, micro_module in my_dispatcher.modules:
                micro_module.visualise_model(realX, realY, "")

if __name__ == "__main__":
    pass
