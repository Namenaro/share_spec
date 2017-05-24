# -*- coding: utf-8 -*
import model_visualiser
import dispatcher


def experiment_1():
    my_dispatcher = dispatcher.Dispatcher(num_modules=5)
    my_visualizer = model_visualiser.Visualizer()
    for i in range(100):
        my_dispatcher.feed_next_data_point_to_modules()
        my_dispatcher.try_consolidation()
        if i%30 == 0:
            for micro_module in my_dispatcher.modules:
                my_visualizer.visualise_model(micro_module)

experiment_1()
