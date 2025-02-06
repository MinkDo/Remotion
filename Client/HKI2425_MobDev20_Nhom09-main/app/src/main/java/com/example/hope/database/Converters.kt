package com.example.hope.database

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.room.TypeConverter
import com.example.hope.reminder.data.RepeatOption
import java.time.LocalDate
import java.time.LocalTime
import java.time.format.DateTimeFormatter

object Converters {

    @RequiresApi(Build.VERSION_CODES.O)
    private val dateFormatter: DateTimeFormatter = DateTimeFormatter.ISO_LOCAL_DATE

    @RequiresApi(Build.VERSION_CODES.O)
    private val timeFormatter: DateTimeFormatter = DateTimeFormatter.ISO_LOCAL_TIME

    // Chuyển đổi LocalDate -> String
    @RequiresApi(Build.VERSION_CODES.O)
    @TypeConverter
    fun fromLocalDate(date: LocalDate?): String? {
        return date?.format(dateFormatter)
    }

    // Chuyển đổi String -> LocalDate
    @RequiresApi(Build.VERSION_CODES.O)
    @TypeConverter
    fun toLocalDate(dateString: String?): LocalDate? {
        return dateString?.let { LocalDate.parse(it, dateFormatter) }
    }

    // Chuyển đổi LocalTime -> String
    @RequiresApi(Build.VERSION_CODES.O)
    @TypeConverter
    fun fromLocalTime(time: LocalTime?): String? {
        return time?.format(timeFormatter)
    }

    // Chuyển đổi String -> LocalTime
    @RequiresApi(Build.VERSION_CODES.O)
    @TypeConverter
    fun toLocalTime(timeString: String?): LocalTime? {
        return timeString?.let { LocalTime.parse(it, timeFormatter) }
    }

    @TypeConverter
    fun fromRepeatOption(value: RepeatOption?): String? {
        return value?.name
    }

    @TypeConverter
    fun toRepeatOption(value: String?): RepeatOption? {
        return value?.let { RepeatOption.valueOf(it) }
    }
}