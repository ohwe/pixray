## BACKLOG

### Первичное
* ~втащить проектор под капот~
* втащить логику сбалансированных градиентов (!)
* пофиксить ширину (!)
* вид почти сверху - больше опрокидывания
* добавить сглаживающий лосс -- считать средний поточечный градиент (torch.gradient) по квадратной матрице высот, штрафовать за резкие перепады
* сглаживание по цветам
* переписать проектор, чтоб можно было крутить сильнее чем в пределах 90 градусов без нарушения порядка отрисовки (менять порядок отрисовки)
* подобрать набор проекций, оптимально защищающий от дурилки картонной 

### Проверить/попробовать
* возможно, добавить премию за глобальный перепад весов (max-min), чтобы стимулировать отойти от плоского
* возможно, инициализировать высоты изначально чем-то вроде sin(x)*cos(y), чтобы стимулировать отойти от плоского
* возможно, кропать крайние ряды, которые шумят
* попробовать сделать лицо -- анфас и профиль
* вытаскивать шов в видимую область (рандомный циклический сдвиг)

### Прекрасное далёко
Тут некоторые мысли, что можно сделать в плюс к обычному вокселятору
* text2voxel
  * https://twitter.com/apeoffire/status/1476676130535051270
  * https://colab.research.google.com/drive/1y0cR5goZ2go6SlYqVIZy7e7g0cOE3prE
  * https://twitter.com/apeoffire/status/1478465287028711427
* https://arxiv.org/abs/2109.12922 + https://twitter.com/NJetchev
* https://paperswithcode.com/paper/text2mesh-text-driven-neural-stylization-for?from=n22
* https://twitter.com/b_nicolet/status/1468954052524273664
* https://twitter.com/ajayj_/status/1436838891957342209
* https://twitter.com/danielrussruss/status/1436209681265950722
* https://www.tensorflow.org/graphics/api_docs/python/tfg/rendering
* https://revdancatt.com/2020/01/30/penplotting-perlin-landscapes
* https://twitter.com/twominutepapers/status/1458091504174575632
* https://arxiv.org/abs/2109.07161 + https://github.com/saic-mdal/lama
* nerf / https://www.ajayj.com/dietnerf
