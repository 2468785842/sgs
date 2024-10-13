# 上
i_up = r'.\sgs_image\01_up.png'
# 左
i_left = r'.\sgs_image\02_left.png'
# 下
i_un = r'.\sgs_image\03_un.png'
# 右
i_right = r'.\sgs_image\04_right.png'
# 风
i_wind = r'.\sgs_image\05_wind.png'
# 火
i_fire = r'.\sgs_image\06_fire.png'
# 雷
i_ray = r'.\sgs_image\07_ray.png'
# 电
i_electricity = r'.\sgs_image\08_electricity.png'

import os
import sys

LDPath = r"D:\leidian\LDPlayerVK"

# 设置 ADBUTILS_ADB_PATH 环境变量, 雷电模拟器路径因为要用模拟器自带的adb
os.environ['ADBUTILS_ADB_PATH'] = LDPath
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio
import time
import coord
import cv2
import numpy as np
import uiautomator2 as u2

from typing import Optional
from enum import Enum

from PIL.Image import Image

import rwlock

class GameState(Enum):
    RUNNING = 1
    SKILL = 2
    OVER = 3
    LAGAN = 4


stopFlag = threading.Event()

curScreen = None
d: Optional[u2.Device] = None

scrst: Image = None
scrstLock = threading.Lock()
scrstCatcherReady = threading.Event()

controlReady = threading.Event()

laGanTime = 0
laGanTimeRWLock = rwlock.RWLock()

normalSpeed = .06

speed = normalSpeed
speedLock = threading.Lock()

gameState: GameState = None
gameStateRWLock = rwlock.RWLock()


# update screenshot
def ScreenCatcher():
    global scrst

    while not stopFlag.is_set():
        try:
            with scrstLock:
                scrst = d.screenshot()
                assert scrst is not None
        except Exception as e:
            print("Screenshot Failed, Retry: ", e)
            continue

        if not scrstCatcherReady.is_set():
            print('ScreenCatcher Ready!')
            scrstCatcherReady.set()

        # 并不需要更新太频繁
        time.sleep(.04)

    print('ScreenCatcher Stop')


def VigourChecker():
    global speed

    scrstCatcherReady.wait()
    controlReady.wait()

    with gameStateRWLock.r_locked():
        state = gameState

    while state != GameState.OVER and not stopFlag.is_set():
        newSpeed = normalSpeed if checkColorRange(
            coord.norm_color, 10, 
            getPixelColor(coord.get_man_color['x'], coord.get_man_color['y'])
        ) else .08 / (normalSpeed + .0075) # 反比例函数normal越小,这个越大

        with speedLock:
            speed = newSpeed

        with gameStateRWLock.r_locked():
            state = gameState

        time.sleep(0.5)


def ScreenControlThread():
    global gameState

    scrstCatcherReady.wait()

    # time.sleep(2)
    # d.click(coord.start_fish['x'], coord.start_fish['y'])
    time.sleep(2)
    # Step 5: 抛竿
    d.swipe(
        coord.start_fish['x'],
        coord.start_fish['y'],
        coord.start_fish['x'], 
        coord.start_fish['y'] - 100, 
        .5
    )
    time.sleep(5.93)
    # Step 6:鱼咬耳了
    d.click(coord.start_fish['x'], coord.start_fish['y'])
    time.sleep(1.2)
    # Step 7:正式钓鱼
    # 模拟按下鼠标左键
    d.long_click(coord.start_fish['x'], coord.start_fish['y'], 1.8)
    # time.sleep(.5)

    print('Running')

    state = GameState.RUNNING
    with gameStateRWLock.w_locked():
        gameState = state

    controlReady.set()

    while state != GameState.OVER and not stopFlag.is_set():
        if state == GameState.SKILL:
            print('致命一击')
            time.sleep(0.1)
            tag = getTagColor(coord)
            
            match tag:
                case 6:
                    clickMatchingTags(coord.six_tag_x)
                case 4:
                    clickMatchingTags(coord.four_tag_x)
                case _: # 5
                    clickMatchingTags(coord.five_tag_x)
                
            state = GameState.OVER
            break
        elif state == GameState.RUNNING:
            # print('点击钓鱼按钮')

            x, y = coord.start_fish['x'], coord.start_fish['y']
            d.click(x, y)
            with speedLock:
                time.sleep(speed)
        elif state == GameState.LAGAN:
            current_gan = True
            with laGanTimeRWLock.r_locked():
                if time.time() - laGanTime < 5:
                    current_gan = False
            if current_gan:
                d.drag(
                    coord.start_fish['x'],
                    coord.start_fish['y'],
                    coord.start_fish['x'],
                    coord.start_fish['y'] - 100, 
                    duration=.2
                )
                with laGanTimeRWLock.w_locked():
                    laGanTime = time.time()
                print("看连续拉了几次杆")
                time.sleep(0.6)

        with gameStateRWLock.r_locked():
            state = gameState

    ## Update State
    with gameStateRWLock.w_locked():
        gameState = state
        
    if stopFlag.is_set(): return

    time.sleep(7)
    print('点击再钓一次')
    d.click(coord.again_fish['x'], coord.again_fish['y'])
    controlReady.clear()


def StateUpdater():
    global gameState

    scrstCatcherReady.wait()

    controlReady.wait()

    color = coord.pole_color
    # 倒计时图标     其中倒计时无,风有则终极一击，都无则退出，倒计时有，凤无则正常
    color1 = coord.countdown_tag_color
    # 风图标
    color2 = coord.test_wind_color
    range_value = 20
    with gameStateRWLock.r_locked():
        state = gameState

    while state != GameState.OVER and not stopFlag.is_set():
 
        target = getPixelColor(
            coord.get_pole_color['x'], coord.get_pole_color['y'])
        new_gan = not checkColorRange(color, range_value, target)

        # 拉杆持续时间大概要5s,我们在5s之后再操作
        with laGanTimeRWLock.r_locked():
            if laGanTime != 0 and state != GameState.LAGAN and new_gan:
                if time.time() - laGanTime < 5:
                    state = GameState.LAGAN

        # 倒计时
        target1 = getPixelColor(
            coord.countdown_tag['x'], coord.countdown_tag['y'])
        # 风
        target2 = getPixelColor(
            coord.test_wind['x'], coord.test_wind['y'])

        if (not checkColorRange(color1, range_value, target1) and
                checkColorRange(color2, range_value, target2)):
            state = GameState.SKILL
        elif (not checkColorRange(color1, range_value, target1) and
            not checkColorRange(color2, range_value, target2)):
            time.sleep(1)
            # 倒计时
            target1 = getPixelColor(
                coord.countdown_tag['x'], coord.countdown_tag['y'])
            # 风
            target2 = getPixelColor(
                coord.test_wind['x'], coord.test_wind['y'])
            if (checkColorRange(color1, range_value, target1) and
                checkColorRange(color2, range_value, target2)):
                state = GameState.SKILL
            else:
                state = GameState.OVER

        with gameStateRWLock.r_locked():
            if gameState == GameState.OVER:
                break

        with gameStateRWLock.w_locked():
            if state != gameState: 
                print('Game State: ', state)
            gameState = state

        time.sleep(1)


async def check_for_exit():
    global stopFlag
    print("按下 'q' 退出程序")
    await asyncio.get_event_loop() \
        .run_in_executor(None, sys.stdin.read, 1)
    stopFlag.set()


# 钓鱼循环次数
runCount = 30
async def initThreads():
    global runCount
    global gameState
    print('start init threads')
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(5) as pool:
        # 启动监测退出的任务
        asyncio.create_task(check_for_exit())
        # 屏幕抓取会一直运行直到退出程序
        loop.run_in_executor(pool, ScreenCatcher)
        while runCount > 0:

            tasks = [
                loop.run_in_executor(pool, ScreenControlThread),
                loop.run_in_executor(pool, StateUpdater),
                loop.run_in_executor(pool, VigourChecker),
            ]

            while not all(task.done() for task in tasks):
                await asyncio.sleep(1.5)
                if stopFlag.is_set():
                    # 模拟器好像有内存泄漏?(内存占用飙升)
                    # 也有可能是 uiautomator 的问题??
                    d.stop_uiautomator()
                    print("正在结束进程(请等待几秒钟)!!!")
                    exit()

            with gameStateRWLock.w_locked():
                gameState = None

            runCount -= 1


async def initEnumlator():
    """
    1. 检查模拟器是否存在
    2. 尝试连接adb
    3. 等待打开三国杀应用
    """
    global d
    try:
        d = u2.connect()
    except Exception as e:
        print('Connect Enumlator Failed: ', e)
        exit()

    print('Connect Success! Enumlator info: \n', d.info)
    
    # check current app info
    app_cur = d.app_current()['package']

    if app_cur is None and app_cur.find('.sgs.') == -1:
        print('你需要启动三国杀!')
        exit()
    else:
        print('已经打开三国杀移动版')
    print('注意必须先!!! 打开活动界面, 点击开始钓鱼, 进入抛竿页面')
       

def checkColorRange(color, range_value, target):
    """
    检查给定的颜色值是否在指定的范围内。

    参数:
    color -- 一个包含r, g, b值的元组。
    range_value -- 允许的颜色值与目标值之间的最大差异。
    r_target, g_target, b_target -- 目标颜色的r, g, b值。

    返回:
    如果颜色在指定范围内，返回True；否则返回False。
    """
    return all(abs(c - t) <= range_value for c, t in zip(color, target))


def getPixelColor(x, y, retryCount = 5):
    if retryCount == 0:
        raise OSError('loading image failed')

    try:
        with scrstLock:
            img = scrst.convert('RGB')
    except OSError as e:
        print(f"Error loading image retry: {e}")
        img = getPixelColor(x, y, retryCount - 1)

    return img.getpixel((x, y))


# 通过图片,判断当前界面是否是我们想要的界面
# yuan_path 原准备对比图,cut_position 准备截取坐标{start_x,start_y,width,height},threshold 置信度
def getImageExist(yuan_path, cut_position, threshold):
    # 读取图像
    image = cv2.imread(yuan_path)
    # 转为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 增强对比度
    enhanced_image = cv2.equalizeHist(gray_image)
    # 获取窗口位置
    region = (
        cut_position['start_x'], 
        cut_position['start_y'],
        cut_position['width'] + cut_position['start_x'],
        cut_position['height'] + cut_position['start_y'])  # 替换为实际的截图区域
    npArr = None
    # 获取需要截图的区域
    with scrstLock:
        npArr = np.array(scrst.crop(region))
    # 转为灰度图像
    gray_get_image = cv2.cvtColor(npArr, cv2.COLOR_BGR2GRAY)
    # 增强对比度
    enhanced_get_image = cv2.equalizeHist(gray_get_image)
    # 模板匹配
    result = cv2.matchTemplate(enhanced_get_image, enhanced_image, cv2.TM_CCOEFF_NORMED)
    # 设定置信度阈值
    loc = np.where(result >= threshold)
    # 计算模板尺寸
    template_height, template_width = enhanced_image.shape
    # 检查是否找到匹配区域
    if len(loc[0]) > 0:
        # 获取匹配区域的中心位置
        i, all_x, all_y = 0, 0, 0
        for pt in zip(*loc[::-1]):  # 反转行列坐标
            x, y = pt
            # 计算中心位置
            all_x += x + template_width // 2
            all_y += y + template_height // 2
            i = i + 1
        c_x = round(all_x / i)
        c_y = round(all_y / i)
        # 返回结果为相对，雷电模拟器的坐标
        last_x = cut_position['start_x'] + c_x
        last_y = cut_position['start_y'] + c_y
        return last_x, last_y
    else:
        return -1, -1


# 匹配tag，返回后续要点击的坐标, 顺序点击
def clickMatchingTags(arr):
    # 定义要匹配的图像和对应的坐标
    tags = [
        (i_up, coord.up),
        (i_left, coord.left),
        (i_un, coord.un),
        (i_right, coord.right),
        (i_wind, coord.wind),
        (i_fire, coord.fire),
        (i_ray, coord.ray),
        (i_electricity, coord.electricity)
    ]
    
    temp_arr = []
    for x in arr:
        # 等待被截图坐标
        cut_cor = {
            'start_x': x - 55,
            'start_y': coord.all_tag_y - 55,
            'width': 110,
            'height': 110
        }

        for img, tag_coord in tags:
            temp_x, temp_y = getImageExist(img, cut_cor, 0.9)
            if temp_x != -1 and temp_y != -1:
                temp_arr.append(tag_coord)
                break  # 找到匹配后跳出当前循环

    # 点击找到的坐标
    for data in temp_arr:
        d.click(data['x'], data['y'])
        time.sleep(0.2)


# 判断颜色
def getTagColor(coord):
    if checkColorRange(coord.center_color, 20, getPixelColor(coord.center_tag['x'], coord.center_tag['y'])):
        if checkColorRange(coord.four_six_color, 20, getPixelColor(coord.four_six_tag['x'], coord.four_six_tag['y'])):
            return 6
        else:
            return 4
    else:
        return 5


async def main():
    await initEnumlator()
    await initThreads()


if __name__ == "__main__":
    asyncio.run(main())
