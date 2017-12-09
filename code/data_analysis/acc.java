private float[] gravity = new float[3];   //重力在设备x、y、z轴上的分量
private float[] motion = new float[3];  //过滤掉重力后，加速度在x、y、z上的分量
private double ratioY; 
private double angle; 
private int counter = 1; 

public void onSensorChanged(SensorEvent event) {  
    for(int i = 0 ; i < 3; i ++){ 
        /* accelermeter是很敏感的，看之前小例子的log就知道。因为重力是恒力，我们移动设备，它的变化不会太快，不象摇晃手机这样的外力那样突然。因此通过low-pass filter对重力进行过滤。这个低通滤波器的权重，我们使用了0.1和0.9，当然也可以设置为0.2和0.8。 */
        gravity[i] = (float) (0.1 * event.values[i] + 0.9 * gravity[i]); 
        motion[i] = event.values[i] - gravity[i]; 
    } 
    
    //计算重力在Y轴方向的量，即G*cos(α) 
    ratioY = gravity[1]/SensorManager.GRAVITY_EARTH; 
    if(ratioY > 1.0) 
        ratioY = 1.0; 
    if(ratioY < -1.0) 
        ratioY = -1.0; 
    //获得α的值，根据z轴的方向修正其正负值。 
    angle = Math.toDegrees(Math.acos(ratioY)); 
    if(gravity[2] < 0) 
        angle = - angle; 
    
    //避免频繁扫屏，每10次变化显示一次值 
    if(counter ++ % 10 == 0){ 
        tv.setText("Raw Values : \n" 
                +  "   x,y,z = "+ event.values[0] + "," + event.values[1] + "," + event.values[2] + "\n"
                +  "Gravity values : \n" 
                +  "   x,y,z = "+ gravity[0] + "," + gravity[1] + "," + gravity[2] + "\n"
                +  "Motion values : \n" 
                +  "   x,y,z = "+ motion[0] + "," + motion[1] + "," + motion[2] + "\n"
                +  "Y轴角度 :" + angle    ); 
        tv.invalidate(); 
        counter = 1; 
    }     
} 