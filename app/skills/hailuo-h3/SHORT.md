# СКИЛЛ «MiniMax H3»: как писать промпт (краткая версия для Агента)

## Ограничения
- Промпт не длиннее 7000 символов. Клип 5-15 секунд, 24 кадра в секунду, звук (стерео) генерируется всегда.
- Референсы: до 9 картинок, 3 видео, 3 аудио. Веса `(слово:1.3)` не поддерживаются, приоритет задаётся только словами.
- Без имён реальных людей, знаменитостей и персонажей франшиз (Marvel, Disney и т.п.): их фильтр блокирует. Предложи оригинального героя с похожим видом.
- Кровь, раны, насилие: не пиши физиологичные детали как действие, передавай угрозу дистанцией, взглядом и позой. Близость: герои одеты, напряжение через кадрирование и реакции окружения.

## Формат: три поля, всегда в таком порядке
```
[строка выравнивания референсов: только если есть картинки, потом пустая строка]

integrated_multimodal_description: [Shot 1] <стиль, стартовая композиция, герои, действие, камера, реплики>

overall_soundscape: <фоновые звуки сцены, 1-4 предложения, без диалогов и музыки>

non_diegetic_music: <инструменты, темп, динамика, 1-3 предложения, или N/A>
```
- Не один слитный абзац, а именно три поля. Не мешать звук среды и музыку. Не просить музыку в одном месте и `N/A` в другом.
- `integrated_multimodal_description` - главная линия времени. Порядок мысли: стиль (например `Live-action, cinematic`) -> стартовая композиция -> действие по нарастающей -> результат или реакция. Первый шот без таймкода: `[Shot 1] ...`. Следующие: `[Shot 2] At 00:04.500, the camera cuts to ...`, таймкод строго по возрастанию и внутри длины клипа. Новый шот только когда зритель получает новое (место, ракурс, момент); простое приближение это движение камеры, не новый шот.
- Нужно наблюдаемое действие и изменение (триггер, промежуточное состояние, итог): описание одного внешнего вида даёт мёртвую картинку.
- `overall_soundscape`: шаги, ветер, дождь, дыхание, шелест ткани, предметы. `non_diegetic_music`: только инструменты, темп, ритм, динамика, без слов вроде «драматично» (эмоция передаётся музыкальной фактурой).
- Всё на английском, реплики дословно на языке оригинала.

## Камера
Тип движения + амплитуда + скорость, вписанные в естественное предложение (не списком в конце): Push In / Pull Out, Zoom In / Out, Pan Left / Right, Truck Left / Right, Tilt Up / Down, Pedestal Up / Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly, POV, Roll. Уточнение `with small / large amplitude`, `at slow / fast speed` только где важно (среднюю амплитуду и обычную скорость не пишут). Не сваливай в один шот несовместимые движения камеры.

## Диалог
- У каждого говорящего устойчивый ID: `(S1)`, `(S2)`, вместе `(S1,S2)`. При первом появлении кратко: возраст, тембр, темп речи (вне блока).
- Реплика дословно внутри `<d>[Язык] текст</d>`, без перевода и правок.
- Голос за кадром: `says in an off-screen voiceover` и обязательно указать, что губы персонажа закрыты.
- Текст в кадре (вывеска, титр) в прямых кавычках дословно. Идеальной типографики не ждать.

## Строка выравнивания (когда есть картинки, ставится первой, потом пустая строка)
- Одна стартовая картинка: `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
- Первый и последний кадр: `How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.` Тело описывает ПУТЬ между кадрами.
- Только финальный кадр: то же с `<Picture 1> (from [Shot N]) aligns with the S.SS-second mark`, начало придумай правдоподобное.
- Нет картинок (только текст): строки выравнивания нет.

## Несколько референсов (Omni-Reference)
Теги `<Picture N>`, `<Video N>`, `<Audio N>` (номера отдельно для каждого типа, в порядке слотов); их ставишь ты, пользователь их не пишет. Каждому файлу ОДНА явная роль: `<Picture 1> is the character reference (keep this face exactly). <Picture 2> is the clothing reference. <Video 1> is a camera-movement reference only, take no faces or people from it. <Audio 1> is the background sound reference.` Роли: character, object, scene, keyframe, voice (тембр), style, composition, motion, camera. Если есть и картинка с лицом, и видео, лицо берётся только с картинки. Внешность героя с картинки словами не расписывай.

## Частые ошибки
- Красивая неподвижная картинка: нет действия, добавь триггер, изменение и итог.
- Камера дёргается: несколько команд свалено в конце, оставь одну естественную фразу.
- Герой говорит не то: реплика пересказана, нужна дословная в `<d>` и `(S1)`.
- При двух картинках «прыжки»: назвал только точки, опиши путь между ними.
- Текст из Word с невидимыми символами даёт ошибку: вставлять как простой текст.

## Пример (текст, 8 секунд)
```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a close medium shot frames a white ceramic cup on a wooden windowsill at early morning, soft golden light from the left. The camera pushes in with small amplitude at slow speed as steam rises in slow curling trails. A hand enters the frame and wraps around the cup. [Shot 2] At 00:05.000, the camera cuts to a close-up of the coffee surface as a spoon stirs it once, sending a small ripple across the crema.

overall_soundscape: A quiet kitchen ambience with a distant clock tick and the soft clink of a spoon against ceramic.

non_diegetic_music: A single sustained piano chord at a slow tempo, joined by a soft acoustic guitar pattern that fades in gradually.
```
