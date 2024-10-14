### 修改自 [](https://github.com/siriusnouers/sgs)

## 为什么?
1. 将原来的pyautogui库实现自动化变为uiautomator2实现自动化
2. 获取屏幕信息截图改为uiautomator2
3. 修改代码架构

### 因为pyautogui是在操作系统端进行鼠标点击和屏幕截图不能后台运行

### 而uiautomator2使用adb在模拟器中安装一个服务
1. 可以直接调用模拟器中Android系统截图, 
2. 点击操作使用的是Android系统自带无障碍服务
3. 通信使用的rpc
4. 可以后台运行

## 怎么用?
1. 不需要修改Windows系统的任何东西(因为直接在模拟器中操作, 不需要系统窗口相对偏移和模拟器标题栏高度)
2. 基本支持所有模拟器(如果你的真机是1600x900分辨率也不是不可以支持)
3. 只要保证模拟器分辨率为`1600x900`即可
4. 需要使用模拟器的`adb`功能
5. 安装py库`pip install uiautomator2 readerwriterlock opencv`
6. 修改代码中的`EmulatorPath = r"D:\Program Files\Netease\MuMu Player 12\shell"`为你自己的模拟器安装路径(下方必须有adb.exe必须支持adb)
8. 运行`python ./sgs_automatic_uiautomator2.py`之后输入循环次数即可

 ## 问题?
 1. 可能遇到`Ctrl+c`或输入`Enter`无法退出的情况, 需要直接关闭终端命令行
 2. 终结一击好像识别不了有点问题?(不是传奇鱼基本可以成功...不影响), PS: 因为传奇鱼少我没注意
 3. 无
