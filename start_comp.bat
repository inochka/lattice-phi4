$action = New-ScheduledTaskAction -Execute 'N:\PycharmProjects\lattice-phi4-master\.venv\Scripts\python.exe' -Argument 'N:\PycharmProjects\lattice-phi4-master\hmc_multiprocessing.py'
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddHours(1)
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "RunPythonScript" -Description "Запуск Python скрипта через 1 час"