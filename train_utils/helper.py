from    datetime    import  timedelta


__all__: list[str] = ['sec_to_hms', 'hms_to_sec']


def sec_to_hms(seconds: float) -> str:
    return str(timedelta(seconds=seconds))

def hms_to_sec(hours: int, minutes: int, seconds: int) -> int:
    return int(timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds())
