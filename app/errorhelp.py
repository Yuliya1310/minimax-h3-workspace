"""Plain-words explanation of errors for the user: what happened and what to do.

Every error that goes to the screen passes through `explain()`. A known error gets «Что случилось» and «Что делать» in front, the technical
text stays below (so it can be sent to the author). An unknown error that is written in English gets a general advice. Add new cases to RULES
when users send new errors: (regex, what happened, what to do)."""
import re

RULES = [
    (r"Torch not compiled with CUDA",
     "В ComfyUI стоит PyTorch без поддержки видеокарты (так бывает после обновления).",
     "Откройте «Настройки» → «Движок» → «Починить видеокарту (PyTorch)» или нажмите такую же кнопку на главном экране. Нужен интернет, модели не затрагиваются."),
    (r"out of memory|OutOfMemory|Allocation on device|CUDA error: out of memory|not enough memory|Could not allocate|Failed to allocate|cudaMalloc",
     "Не хватило видеопамяти: задание слишком тяжёлое для вашей видеокарты.",
     "Закройте другие программы, которые используют видеокарту (игры, браузер с видео), нажмите «Освободить память» и повторите. Если опять не вышло: уменьшите размер или длительность, выберите профиль полегче."),
    (r"\[WinError 1455\]|paging file|страничн",
     "Не хватило оперативной памяти или файла подкачки Windows.",
     "Закройте лишние программы. Увеличьте файл подкачки Windows (на диске, где есть свободное место) и перезагрузите компьютер."),
    (r"No space left|not enough space|Errno 28|недостаточно места|disk full",
     "На диске закончилось место.",
     "Освободите место на диске, где лежат модели и результаты (нужны десятки гигабайт), и повторите."),
    (r"3D[\\/]+[\w.\-]*(birefnet|dino|trellis|triposplat|flux2)[\w.\-]*.{0,40}not in|(birefnet|dino_v3|triposplat|trellis_2).{0,80}not in \[",
     "ComfyUI не видит файлы моделей для 3D: их ещё нет в нужных папках, или после скачивания ComfyUI не перезапускали.",
     "Откройте страницу «3D»: вверху нажмите кнопку скачивания моделей и дождитесь конца. Потом полностью перезапустите ComfyUI (закройте и откройте) и повторите. Если ComfyUI свой, положите файлы в указанные на странице папки models."),
    (r"value_not_in_list|not in \[|Value not in list|не найден файл модел|model.*not found|No such file.*(safetensors|gguf|ckpt)",
     "ComfyUI не нашёл нужный файл модели: его нет или он лежит не в той папке.",
     "Откройте «Настройки» → блок моделей: там видно, какого файла не хватает. Нажмите «Скачать» или выберите свой файл через «Заменить…», затем обновите страницу."),
    (r"Node .* not found|missing_node_type|unknown node|class_type.*not found|нода.*не найден",
     "Не хватает дополнения (ноды) ComfyUI.",
     "Откройте «Настройки» и нажмите «Докачать недостающие ноды», затем перезапустите ComfyUI. Если ComfyUI свой: установите ноды из подсказки в настройках."),
    (r"shape|size mismatch|mat1 and mat2|state_dict|missing key|unexpected key|dtype",
     "Файл модели не подходит к этой функции (другая версия или другой формат).",
     "Проверьте модели в «Настройках»: для нужной строки нажмите «Заменить…» и выберите файл «как по умолчанию»."),
    (r"CERTIFICATE_VERIFY_FAILED|SSL|ssl:|certificate",
     "Не удалось установить защищённое соединение: мешает антивирус, VPN или неверное время на компьютере.",
     "Проверьте дату и время Windows, временно отключите VPN или «HTTPS-сканирование» в антивирусе и повторите."),
    (r"Cannot connect to host|ConnectionError|Connection refused|Name or service|getaddrinfo|timed out|TimeoutError|ClientConnector|Нет соединения|нет интернета|WinError 10061|WinError 10060",
     "Нет связи: интернета нет, сервис недоступен или ComfyUI закрылся.",
     "Проверьте интернет. Если не помогло, посмотрите вверху главного экрана: там написано, запущен ли ComfyUI и что делать. Для облака и арендованных серверов попробуйте отключить или включить VPN."),
    (r"(HTTP|status|code|error)\D{0,8}401|Unauthorized|invalid api key|Incorrect API key|invalid_api_key|authentication",
     "Сервис не принял ключ доступа (API-ключ неверный, не активен или закончился срок).",
     "Откройте «Настройки», проверьте ключ (без пробелов в начале и конце), при необходимости создайте новый в личном кабинете сервиса."),
    (r"(HTTP|status|code|error)\D{0,8}402|insufficient|quota|balance|billing|credit|Payment Required|недостаточно средств",
     "На счёте сервиса недостаточно средств или исчерпан лимит.",
     "Пополните баланс в личном кабинете сервиса (RunningHub, RunPod или API) и повторите."),
    (r"(HTTP|status|code|error)\D{0,8}429|rate limit|Too Many Requests",
     "Сервис получил слишком много запросов подряд.",
     "Подождите минуту-две и повторите. Не запускайте много заданий одновременно."),
    (r"(HTTP|status|code|error)\D{0,8}403|Forbidden|Cloudflare|blocked",
     "Сервис отказал в доступе: часто это из-за VPN или региона.",
     "Выключите VPN (или, наоборот, включите другой) и повторите. Если сервис ваш собственный, проверьте логин и пароль."),
    (r"Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout",
     "Сервис временно не работает (ошибка на его стороне).",
     "Это не ваша ошибка. Подождите несколько минут и повторите."),
    (r"Permission denied|PermissionError|Access is denied|WinError 5\b|Отказано в доступе",
     "У программы нет прав записать или прочитать файл: папка защищена, или файл открыт в другой программе.",
     "Закройте программу, которая использует файл (плеер, проводник), выберите другую папку для результатов или запустите программу не из «Program Files»."),
    (r"ffmpeg|Invalid data found|moov atom|Error while decoding|could not find codec",
     "Не удалось прочитать или записать видео: файл повреждён или в нём необычный формат.",
     "Попробуйте другой файл или пересохраните видео в MP4 (H.264) любым конвертером, затем загрузите снова."),
    (r"PyTorch|CUDA.*not available|no kernel image|CUDA driver|driver version|cuda.*initialization",
     "Проблема с драйвером видеокарты или с её поддержкой.",
     "Обновите драйвер NVIDIA (nvidia.com/drivers), перезагрузите компьютер и повторите."),
    (r"No module named|ModuleNotFoundError|ImportError",
     "ComfyUI не хватает одной из библиотек Python.",
     "Нажмите «Обновить ComfyUI» в «Настройках» и перезапустите программу. Если не помогло, нажмите «Отчёт для разработчика» и пришлите файл автору."),
]
_RULES = [(re.compile(p, re.I), a, b) for p, a, b in RULES]

GENERAL_WHAT = "Что-то пошло не так, и программа не смогла выполнить задание."
GENERAL_DO = ("Повторите ещё раз. Если ошибка осталась: нажмите «Освободить память», перезапустите программу и повторите. "
              "Если опять не получилось, откройте «Настройки» → «Отчёт для разработчика» и пришлите файл автору вместе с текстом ниже.")


def explain(message):
    """The message with a plain explanation in front (or unchanged when it is already explained / is plain Russian)."""
    text = str(message or "").strip()
    if not text or "Что делать:" in text:
        return text
    for rx, what, todo in _RULES:
        if rx.search(text):
            return f"{what}\nЧто делать: {todo}\n\nТехническая причина: {text}"
    if not re.search("[А-Яа-яЁё]", text):   # an unknown error in English: nothing in it helps a beginner
        return f"{GENERAL_WHAT}\nЧто делать: {GENERAL_DO}\n\nТехническая причина: {text}"
    return text
