import os
from pathlib import Path

import pandas as pd
import wfdb

from .config import RECORDS_FOLDER
from .models import Subject


def get_available_patients(path):
    ids = set()
    for subject in os.listdir(path):
        if os.path.isfile(os.path.join(path, subject)):
            subject_id, _ = os.path.splitext(subject)
            if len(subject_id) == 3:
                ids.add(subject_id)
    return sorted(ids)


get_available_pacients = get_available_patients


def read_subject_info(subject, records_folder=RECORDS_FOLDER):
    data_path = str(Path(records_folder) / subject)
    header = wfdb.rdheader(data_path)

    comments = header.comments[0].split(" ")
    age = int(comments[0])
    gender = comments[1]
    medicine = header.comments[1]

    return [subject, age, gender, medicine]


def read_subject_signal(subject, records_folder=RECORDS_FOLDER):
    data_path = str(Path(records_folder) / subject)
    record = wfdb.rdrecord(data_path)
    return record.p_signal, record.fs, record.sig_name


def read_subject_annotation(subject, records_folder=RECORDS_FOLDER):
    data_path = str(Path(records_folder) / subject)
    return wfdb.rdann(data_path, "atr")


def read_subject_data(subject, records_folder=RECORDS_FOLDER, verbose=False):
    signal, fs, sig_name = read_subject_signal(subject, records_folder)
    annotation = read_subject_annotation(subject, records_folder)
    if verbose:
        print(sig_name)
        print("Signal shape:", signal.shape)
        print("Signal frequency: ", fs)
        print("number of annotated beats: ", len(annotation.sample))
    return signal, fs, annotation


def load_patient_table(records_folder=RECORDS_FOLDER):
    data = [
        read_subject_info(patient, records_folder)
        for patient in get_available_patients(records_folder)
    ]
    return pd.DataFrame(data, columns=["ID", "Idade", "Gênero", "Medicamentos"])


def load_all_subjects(records_folder=RECORDS_FOLDER):
    subjects = []
    for patient in get_available_patients(records_folder):
        subject_number, age, gender, meds = read_subject_info(patient, records_folder)
        signal, fs, notation = read_subject_data(patient, records_folder)
        subjects.append(Subject(subject_number, age, gender, meds, signal, notation, fs=fs))
    return subjects
