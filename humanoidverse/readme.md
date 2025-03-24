正常启动主程序
```shell
roslaunch humanoid_controllers load_kuavo_mujoco_sim.launch
```
无视觉上楼梯指令：其中`--step`参数代表要登上的楼梯数量
```
rosrun humanoid_controllers stairClimbPlanner.py --step 4
```
有视觉上楼梯指令：其中 `--step` 参数代表要登上的楼梯数量，`--error`代表视觉识别楼梯的一个恒定位置偏差
```
rosrun humanoid_controllers visionStairClimbPlanner.py --step 3 --error 0.1
```
视觉部分的启动参考`kauvo-terrain`仓库的`readme.md`需要在上位机启动`docker`并且开启多个节点，具体细节小康知道。

需要注意的点：
- 上楼梯分支的单步控制接口发送的是世界坐标系下的位置，所以不能重复运行`rosrun humanoid_controllers stairClimbPlanner.py --step 1`四次来登上四级台阶
- 使用视觉上楼梯理应先开启视觉模块，然后在终端`rostopic echo`楼梯位置话题的数据，然后量一下实际机器人脚底距离楼梯中心的距离，然后看一下误差，这个误差就是`--error`的值，最后执行命令
- 采用盲走上楼梯需要将机器人放在楼梯前面，具体可以观看`stairClimbPlanner.py`中的代码