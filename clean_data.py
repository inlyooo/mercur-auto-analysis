"""Очистка данных о продажах автомобилей в Казахстане за 2019 год."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def clean_data(source):
    df = pd.read_csv(source, sep=";", decimal=",", encoding="utf-8", low_memory=False)
    print(f"Исходные данные: {len(df)} строк, {len(df.columns)} столбцов")
    empty = df.isna().all(axis=1).sum()
    df = df.dropna(how="all")
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates().copy()
    print(f"Удалено пустых строк: {empty}, полных дубликатов: {duplicates}")

    # Сначала убираю лишние пробелы, чтобы словари находили все варианты
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].str.strip().str.replace(r"\s+", " ", regex=True)
        df[column] = df[column].replace({"#Н/Д": pd.NA, "": pd.NA})

    # У Renault часть значений попала в соседние столбцы
    shifted = df["Бренд"].eq("Renault") & df["Вид топлива"].isin(["1,6", "2"])
    df.loc[shifted, "Коробка передач"] = df.loc[shifted, "Объём двиг, л,"].values
    df.loc[shifted, "Объём двиг, л,"] = df.loc[shifted, "Вид топлива"].values
    df.loc[shifted, "Вид топлива"] = pd.NA
    print(f"Исправлено строк со сдвигом полей Renault: {shifted.sum()}")

    # У Niva есть правильные 1,7 л, а рядом ошибочный ряд от 2,7 до 26,7
    niva = df["Бренд"].eq("Chevrolet") & df["Модель"].eq("Niva")
    niva_fixed = (niva & df["Объём двиг, л,"].ne("1,7")).sum()
    df.loc[niva, "Объём двиг, л,"] = "1,7"

    # Для этой модификации Polo остальные записи содержат 1,6 л
    polo = df["Бренд"].eq("Volkswagen") & df["Модель"].eq("Polo")
    polo_error = polo & df["Объём двиг, л,"].eq("8,7")
    df.loc[polo_error, "Объём двиг, л,"] = "1,6"
    print(f"Исправлен объём двигателя: Niva — {niva_fixed}, Polo — {polo_error.sum()}")

    countries = {
        "Германия": "DEU", "США": "USA", "Австрия": "AUT",
        "Республика Казахстан": "KAZ", "Российская Федерация": "RUS",
        "Корея": "KOR", "Япония": "JPN", "Таиланд": "THA", "Китай": "CHN",
        "UK": "GBR", "Узбекистан": "UZB", "Венгрия": "HUN", "Турция": "TUR",
        "Испания": "ESP", "Нидерланды": "NLD", "Польша": "POL",
        "Швеция": "SWE", "Белоруссия": "BLR", "Бельгия": "BEL",
    }
    df["Страна-производитель"] = df["Страна-производитель"].map(countries)
    fuel = {"бензин": "F", "дизель": "D", "электро": "E", "электричество": "E", "гибрид": "HYB"}
    df["Вид топлива"] = df["Вид топлива"].str.lower().map(fuel)
    # В четырёх строках гибрид указан в модификации, но не в поле топлива
    hybrid = df["Модификация"].str.contains(r"E-Hybrid", case=False, na=False)
    df.loc[hybrid, "Вид топлива"] = "HYB"

    engine_text = df["Объём двиг, л,"].str.replace(",", ".", regex=False)
    df["Объём двиг, л,"] = pd.to_numeric(
        engine_text.str.extract(r"^(\d+(?:\.\d+)?)")[0], errors="coerce"
    )
    # 400 л.с. и 88 kWh — мощность и ёмкость батареи, а не литры
    electric = df["Вид топлива"].eq("E")
    df.loc[electric, "Объём двиг, л,"] = 0.0
    df.loc[(~electric) & df["Объём двиг, л,"].eq(0), "Объём двиг, л,"] = np.nan

    # Топливо восстанавливаю только по однозначному сочетанию марки, модели и объёма
    keys = ["Бренд", "Модель", "Объём двиг, л,"]
    known_fuel = df.dropna(subset=keys + ["Вид топлива"]).groupby(keys)["Вид топлива"]
    fuel_lookup = known_fuel.first()[known_fuel.nunique().eq(1)]
    missing_fuel = df["Вид топлива"].isna()
    fuel_values = pd.MultiIndex.from_frame(df.loc[missing_fuel, keys]).map(fuel_lookup)
    df.loc[missing_fuel, "Вид топлива"] = fuel_values.to_numpy()
    print(f"Топливо восстановлено по однозначным аналогам: {(missing_fuel & df['Вид топлива'].notna()).sum()}")

    drives = {
        "передний": "FWD", "передний (ff)": "FWD", "fwd": "FWD", "ff": "FWD",
        "задний": "RWD", "rwd": "RWD", "полный": "4WD", "awd": "4WD",
        "quattro": "4WD", "4motion": "4WD", "4wd": "4WD", "4 wd": "4WD",
        "4x4": "4WD", "4х4": "4WD", "2wd": "2WD", "2 wd": "2WD",
        "4x2": "2WD", "4х2": "2WD", "4х2.2": "2WD",
    }
    df["Тип привода"] = df["Тип привода"].str.lower().map(drives)

    def transmission(value):
        if pd.isna(value):
            return pd.NA
        value = value.upper().replace(" ", "").replace("/", "")
        if value in ["МЕХ.", "МКПП", "МКП"]:
            return "Механика"
        if value == "РЕДУКТОР":
            return "Автомат"
        value = value.translate(str.maketrans({"А": "A", "М": "M", "Т": "T"}))
        if value in ["0", "4WD", "ПЕРЕДНИЙ", "TDI"]:
            return pd.NA
        if value in ["MT", "5M", "5MT", "6MT"]:
            return "Механика"
        # Робот, вариатор и редуктор электромобиля относятся к автоматическим
        if any(part in value for part in ["AT", "AКП", "TRONIC", "STEPTRONIC", "CVT", "DCT", "DSG", "PDK", "POWERSHIFT"]) or value in ["AMT", "8", "6A", "8A"]:
            return "Автомат"
        return pd.NA

    df["Коробка передач"] = df["Коробка передач"].map(transmission)
    companies = {
        "Mercur Auto": "Меркур Авто", "Mercur Autos": "Меркур Авто",
        "Caspian Motors": "Каспиан Моторс", "Autokapital": "Автокапитал",
        "MMC RUS": "ММС Рус", "Ravon Motors Kazakstan": "Равон Моторс Казахстан",
        "Hino Motors": "Хино Моторс Казахстан",
    }
    df["Компания"] = df["Компания"].replace(companies)
    for column in ["Регион", "Область"]:
        df[column] = df[column].str.title()
    df["Область"] = df["Область"].str.replace(r"^Г\.", "г. ", regex=True)

    for column in ["Количество", "Цена, USD", "Продажа, USD"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # Пропущенное количество при нулевой выручке и положительной цене равно нулю
    missing_quantity = df["Количество"].isna() & df["Продажа, USD"].eq(0) & df["Цена, USD"].gt(0)
    df.loc[missing_quantity, "Количество"] = 0
    df["Количество"] = df["Количество"].astype("Int64")
    df["Год выпуска"] = pd.to_numeric(df["Год выпуска"].str.replace(" ", "", regex=False), errors="coerce").astype("Int64")

    months = {"Январь": 1, "Февраль": 2, "Март": 3, "Апрель": 4, "Май": 5, "Июнь": 6, "Июль": 7, "Август": 8, "Сентябрь": 9}
    df["sale_date"] = pd.to_datetime({"year": df["Год"], "month": df["Месяц"].map(months), "day": 1}) + pd.offsets.MonthEnd(0)
    df = df.drop(columns=["Год", "Месяц", "Наименование дилерского центра", "Форма расчета", "Сегмент", "Сегментация Eng", "Локализация производства", "Тип клиента", "Модификация"])
    df = df.rename(columns={
        "Компания": "company", "Бренд": "brand", "Модель": "model",
        "Год выпуска": "manufacture_year", "Страна-производитель": "country",
        "Вид топлива": "fuel_type", "Объём двиг, л,": "engine_volume",
        "Коробка передач": "transmission", "Тип привода": "drive_type",
        "Регион": "region", "Количество": "quantity", "Цена, USD": "price_usd",
        "Продажа, USD": "revenue_usd", "Область": "area",
        "Сегментация 2013": "segment_2013", "Класс 2013": "class_2013",
    })
    for column in ["fuel_type", "transmission", "drive_type", "segment_2013", "class_2013"]:
        df[column] = df[column].astype("category")
    # После нормализации похожие строки не удаляю: это могут быть разные продажи
    print(f"Готовые данные: {len(df)} строк, {len(df.columns)} столбцов")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default="autokz2019.csv")
    parser.add_argument("--output", default="autokz2019_clean.csv")
    args = parser.parse_args()
    cleaned = clean_data(args.source)
    cleaned.to_csv(Path(args.output), index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    print(f"CSV сохранён: {args.output}")
