package com.skt.spark.r2.ml

object DateUtil {

  import java.text.SimpleDateFormat
  import java.util.Date

  val timeFormat = new SimpleDateFormat("yyyyMMddHHmmss")

  def moveMinutes(time: String, minutes: Int): String = {
    if (minutes == 0) time
    else {
      val date = timeFormat.parse(time)
      timeFormat.format(new Date(date.getTime + minutes * 60000))
    }
  }

  def beforeMinutes(time: String, minutes: Int): String = moveMinutes(time, -1 * minutes)

  def afterMinutes(time: String, minutes: Int): String = moveMinutes(time, minutes)

  def beforeDays(time: String, days: Int): String = {
    if (days == 0) time
    else {
      val date = timeFormat.parse(time)
      timeFormat.format(new Date(date.getTime - days * 24 * 60 * 60000))
    }
  }

  def buildTimeFilter(colName: String, until: String, minDuration: Int, dayCount: Int): String = {
    if (minDuration == 0) {
      (for (i <- 0 until dayCount) yield s"$colName='${beforeDays(until, i)}'").mkString(" OR ")
    } else {
      (for (i <- 0 until dayCount) yield {
        val timeStr = beforeDays(until, i)
        s"($colName>='${beforeMinutes(timeStr, minDuration)}' AND $colName<='$timeStr')"
      }).mkString(" OR ")
    }
  }
}
