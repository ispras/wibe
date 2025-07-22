# WIBE Framework

## Как запустить уже реализованные методы маркирования и атаки на них

1. Клонируем репозиторий и переходим в директорию, куда склонировали (все дальнейшие команды выполняем из этой директории):
```bash
git clone git@gitlab.ispras.ru:watermarking/img-watermarking-test.git
```

Если доступа к репозиторию нет, можно скачать архив с кодом по ссылке (она у вас должна быть), распаковать его и перейти в директорию, куда распаковали

2. Актуализируем сабмодули:
```bash
git submodule update --init --recursive
```

3. Создаем виртуальное окружение и активируем его (для разных ОС это делается немного по-разному, вы знаете, как это сделать):
```bash
python -m venv venv
```

4. Загружаем веса предобученных моделей по ссылке:
```bash
(venv) python download_models.py
```

5. Устанавливаем зависимости:
```bash
(venv) python install_requirements.py
```

6. Определяем переменную окружения с именем **HF_TOKEN** и значением токена для *HuggingFace*, после выполняем вход:
```bash
(venv) python huggingface_login.py
```

Для аутентификации *huggingface_hub* требует токен, сгенерированный на [странице](https://huggingface.co/settings/tokens):
- Добавьте разрешение "Read access to contents of all public gated repos you can access" -- "Чтение содержимого всех доступных закрытых репозиториев, к которым у вас есть доступ".
- Перейдите по [ссылке](https://huggingface.co/black-forest-labs/FLUX.1-dev), запросите доступ и нажмите "Agree and access repository" -- "Согласиться и получить доступ к репозиторию"

7. Запускаем тестирование с заданной конфигурацией:
```bash
(venv) python -m imgmarkbench --config configs/trustmark_msu.yml
```

## Поддерживаемые ОС

- Linux
- Microsoft Windows
