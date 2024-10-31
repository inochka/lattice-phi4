import os

# Получить количество логических ядер (включая гиперпоточность)
logical_cores = os.cpu_count()

# Альтернатива с использованием библиотеки `psutil` для более детального вывода
try:
    import psutil
    physical_cores = psutil.cpu_count(logical=False)  # Физические ядра (без учета гиперпоточности)
    logical_cores_psutil = psutil.cpu_count(logical=True)  # Логические ядра
    print(f"Физические ядра: {physical_cores}, Логические ядра: {logical_cores_psutil}")
except ImportError:
    print(f"Логические ядра (включая гиперпоточность): {logical_cores}")