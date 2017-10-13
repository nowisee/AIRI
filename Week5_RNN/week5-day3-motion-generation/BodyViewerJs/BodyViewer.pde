import processing.opengl.*;
//boolean verbose_=false;

G3DView g3=new G3DView();
Body b=new Body();
int FRAME_RATE = 30;

void setup()
{
    size( 400, 400, P3D );
    frameRate(FRAME_RATE);
    g3.setup(width,height,256);
    b.setup();
}

void draw()
{
    background(255);
    g3.checkMouseWheel();
    g3.draw();
}

void mousePressed()
{
    g3.mousePressed();
}

void mouseDragged()
{
    g3.mouseDragged();
}

void mouseWheel(MouseEvent event)
{
    g3.mouseWheel(event);
}

void g3Objects()
{
    lights();
    g3.translate_center();
    rotateX(PI/2);
    //translate(0,-130,0);
    b.draw();
    float dt=1.0/FRAME_RATE;
    b.update(dt);
}
