# Продажи автомобилей в Казахстане

Курсовая работа по предподготовке и анализу данных за январь — сентябрь 2019 года

Кислицын Дмитрий Геннадьевич, М26-555

- [Отчёт](report.md)
- [Анализ и графики](analysis.ipynb)
- [Очищенный CSV](autokz2019_clean.csv)
- [Скрипт очистки](clean_data.py)

## Запуск

```bash
pip install -r requirements.txt
python clean_data.py autokz2019.csv --output autokz2019_clean.csv
jupyter notebook analysis.ipynb
```

Источник: [материалы задания](https://disk.yandex.ru/d/OmlexqoVSR6mwg)
