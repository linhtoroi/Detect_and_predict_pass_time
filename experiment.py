from src.learn_framework import LFramework

lfr = LFramework(290, 400, 10)
lfr.defineVideoPath(data_name='sangt5', file_extension='.mp4')
# lfr.train(data_name='sangt5',time_data=9)
lfr.load_model()
lfr.run(time_data=9)
