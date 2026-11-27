"""打印当前时间与九九乘法表前三行。"""

from datetime import datetime


def print_current_time() -> None:
    """打印当前本地时间。"""
    current_time = datetime.now()
    print(f"当前时间：{current_time:%Y-%m-%d %H:%M:%S}")


def print_multiplication_table_first_three_rows() -> None:
    """打印九九乘法表的前三行。"""
    print("九九乘法表前三行：")
    for row in range(1, 4):
        expressions = [f"{column}×{row}={column * row}" for column in range(1, row + 1)]
        print("\t".join(expressions))


def main() -> None:
    """按顺序输出时间与乘法表。"""
    print_current_time()
    print_multiplication_table_first_three_rows()


if __name__ == "__main__":
    main()
