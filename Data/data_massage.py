#!/usr/bin/env python3


import pandas
import os.path as path
import io
import math
import numpy


__METADATA_DEFAULT_FILE_PATH__ = path.join(".", "HT_Sensor_metadata.dat")
__DATA_DEFAULT_FILE_PATH__ = path.join(".", "HT_Sensor_dataset.dat")
__METADATA_PICKLED_FILE_PATH__ = path.join(".", "metadata.pkl")
__INDUCTION_ONLY_FILE_PATH__ = path.join(".", "data.pkl")
__FULL_DATA_PICKLED_FILE_PATH__ = path.join(".", "full_data.pkl")


def file_to_string(path):
    # split all lines into tokens, strip whitespace from all tokens, join them all by comma, join all the strings by newline
    return "\n".join(map(",".join, map(str.split, io.open(path, "r").readlines())))


def rebuild_timestamp(sample, metadata):
    import datetime
    import time

    ind = int(sample["induction"])  # induction number
    date = metadata[ind]["date"]  # date
    td = sample["time"]  # fraction of an hour, +-, from the time from the metadata
    t0 = metadata[ind]["t0"]  # stored in hour.fraction of hour after hour that started, which was weird because minutes are a thing

    total_delta_time = t0 + td
    hour = math.floor(total_delta_time)
    fraction_hour = total_delta_time - hour
    minutes = math.floor(fraction_hour * 60.0)
    seconds = math.floor((fraction_hour * 60.0 - minutes) * 60.0)

    # now convert to a uniform dimension so we can do the math that these people should have done
    timestamp_microseconds = (hour * 3.6E9) + (minutes * 60E6) + (seconds * 1E6)

    new_hour = math.floor(timestamp_microseconds / 3.6E9)
    new_minutes = math.floor((timestamp_microseconds % 3.6E9) / 60E6)
    seconds = math.floor(((timestamp_microseconds % 3.6E9) % 60E6) / 1E6)
    us = math.floor(((timestamp_microseconds % 3.6E9) % 60E6) % 1E6)

    string = "{} {} {} {} {}".format(date, hour, minutes, seconds, str(us).zfill(6))

    parser = datetime.datetime.strptime(string, "%m-%d-%y %H %M %S %f")

    return parser.timestamp()


def correct_metadata_delta_times(data, metadata):
    # the metadata stores delta times, but they are incorrectly computed. the final sample for induction 0 has time 2.63, in units fraction of an hour. excluding the samples of the induction that have negative time, there are 2.63 hours of data. ~9400 samples, at 1 Hz, is about 9400 seconds. divided by 3600 to convert to hours, we get about 2.61
    # re-compute the dt values in the metadata
    for ind in metadata.keys():
        induction = data[(data["induction"] == ind) & (data["time"] >= 0.0)]

        # we could have just taken the length of the resultant dataframe, and called that the number of seconds, but the jitter of the measurements would make this inaccurate
        # so take the final computed timestamp, and subtract the beginning
        # the timestamps computed are in seconds + microseconds since the unix epoch, but are already normalized
        true_delta = induction.iloc[len(induction) - 1]["timestamp"] - induction.iloc[0]["timestamp"]
        metadata[ind]["dt_useconds"] = true_delta * 1E6
        metadata[ind]["dt_hours"] = true_delta / 3600

    return None


def summarize_data(metadata, data):
    # time to gas max
    # time to gas mean
    # time to gas median
    # proportion samples above gas median
    # gas std. deviation
    #["time_to_gas_max", "time_to_gas_mean", "time_to_gas_median", "proportion_gas_above_median", "gas_std_deviation"])

    for ind in data["induction"].values:
        #["time_to_gas_max", "time_to_gas_mean", "time_to_gas_median", "proportion_gas_above_median", "gas_std_deviation"])
        induction = data[data["induction"] == ind]
        root_timestamp = induction.iloc[0]["timestamp"]

        # index of where the gas hit its maximal value
        gas_max = induction["R_mean"].argmax()

        # find the time it took for the gas response to hit its peak
        metadata[ind]["time_to_gas_max"] = induction.iloc[induction["R_mean"].argmax()]["timestamp"] - root_timestamp

        gas_max_time = induction.iloc[induction["R_mean"].argmax()]["timestamp"]

    return metadata


def preprocess_data(metadata, data):
    # IF THIS FUNCTION IS CHANGED
    # be sure to remove *.pkl from the directory :(

    # the id column name is confusing, rename it
    metadata.drop(columns="id", inplace=True)
    data.rename(columns={"id": "induction"}, inplace=True)
    metadata.index.names = ["induction"]
    data.index.names = ["id"]
    metadata_dict = metadata.to_dict(orient="index")

    # strangely, there is no induction 95 in the dataset, so delete the key
    del metadata_dict[95]

    # recreate the timestamps from the metadata and the t0 values
    data["timestamp"] = data.apply(lambda sample: rebuild_timestamp(sample, metadata_dict), axis=1)

    # apply the labels for the data, to the data
    data["label"] = data["induction"].apply(lambda ind: metadata_dict[ind]["class"])

    # now that the timestamps are computed, correct the dt measurements in the metadata
    correct_metadata_delta_times(data, metadata_dict)

    data["R_mean"] = data[["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]].apply(numpy.mean, axis=1)

    return metadata, data


def load_data():
    # if the data is already pickled and saved, load it instead of re-doing all the transformations
    # we could try to do some fancy modification time checking to refresh the pickled files
    # instead, it's up to the person modifying this to clear out the cache
    if all(map(path.exists, [__METADATA_PICKLED_FILE_PATH__, __INDUCTION_ONLY_FILE_PATH__ , __FULL_DATA_PICKLED_FILE_PATH__])):
        metadata = pandas.read_pickle(__METADATA_PICKLED_FILE_PATH__)
        full_data = pandas.read_pickle(__FULL_DATA_PICKLED_FILE_PATH__)
        induction_only = pandas.read_pickle(__INDUCTION_ONLY_FILE_PATH__)
    else:
        metadata = pandas.read_csv(io.StringIO(file_to_string(__METADATA_DEFAULT_FILE_PATH__)))
        full_data = pandas.read_csv(io.StringIO(file_to_string(__DATA_DEFAULT_FILE_PATH__)))

        metadata, full_data = preprocess_data(metadata, full_data)

        # retrieve the datasets that are strictly just induction data, not the background control data
        induction_only = full_data[full_data["time"] >= 0.0]

        metadata.to_pickle(__METADATA_PICKLED_FILE_PATH__)
        full_data.to_pickle(__FULL_DATA_PICKLED_FILE_PATH__)
        induction_only.to_pickle(__INDUCTION_ONLY_FILE_PATH__)

    return metadata.to_dict(orient="index"), full_data, induction_only


if __name__ == "__main__":
    metadata, full_data, induction_only = load_data()

    induction_only = induction_only.sample(n=13, random_state=7)

    summarized_data = summarize_data(metadata, induction_only)

    #print(induction_only)
    #print(summarized_data)

    #print(metadata)
    #print(full_data["time"].in())
    #print(induction_only["time"].min())

