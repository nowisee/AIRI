// BodyModel.pde


String   BODY_DATA_FILE  = 
    externals.window.location.hash.substring(1) ||
    externals.canvas.getAttribute('data-processing-argv') ||
    "../body.csv";


class Body {

    float    BODY_SCALE_BASE = 390.0;
    float    SPHERE_DIA_BASE = 10.0;

    int MAX_MOTIONS = 5000;
    int NUM_JOINTS  = 16;
    int NUM_COLS = 3 * NUM_JOINTS + 1;

    /*
     * get_bounds()
     * returns: float[9]
     *  min_x, min_y, min_z, max_x, max_y, max_z, ext_x, ext_y, ext_z
     */
    float[] get_bounds(float[] data, int start_row, int num_rows, int num_joints, int row_stride)
    {

        float[] results = new float[] { 99999.0, 99999.0, 99999.0, -99999.0, -99999.0, -99999.0, 0.0, 0.0, 0.0, };

        for (int c = 0; c < 3; c++) {
            for (int r = 0; r < num_rows; r++) {
                for (int j = 0; j < num_joints; j++) {
                        float x = data[r * row_stride + j * 3 + c];
                        results[0+c] = min(results[0+c], x);
                        results[3+c] = max(results[3+c], x);
                }
            }
            results[6+c] = results[3+c] - results[0+c];
        }

        return results;

    }


    int step = 0;

    float[]initial_joints = new float[NUM_COLS];
    float[]joints = new float[MAX_MOTIONS*NUM_COLS];

    float body_scale;
    float[] body_trans = new float[] {0.0, 0.0, 0.0};
    float sphere_dia;

    int num_motions = 0;

    int[]links = new int[] { //NUM_LINKS * from, to
        10, 9, 9, 8, 8, 13, 8, 14, 13, 12,
        14, 15, 12, 11, 15, 16, 8, 7, 7, 3,
        7, 4, 3, 2, 2, 1, 4, 5, 5, 6,
    };

    void pushStyle_safe() {
        try {
            pushStyle();
        }catch(Exception e){
            //println(e);
        }
    }

    void popStyle_safe() {
        try {
            popStyle();
        }catch(Exception e){
            //println(e);
        }
    }


    void setup()
    {
      String[] lines;
      float[] values;

        lines = loadStrings(BODY_DATA_FILE);

        // println("lines: "+lines.length);
        for (int r = 0; r < lines.length; r++) {
            values = float(lines[r].split(","));
            // println("columns: "+values.length+" num_cols:"+NUM_COLS+" joints:"+joints.length);
            arrayCopy(values, 0, joints, r * NUM_COLS, NUM_COLS);
            num_motions++;
        }
        lines = null;

        // get bounds
        // println("start get bounds");

        // get bounds for first 250 step
        float[] bounds = get_bounds(joints, 0, min(num_motions,500), NUM_JOINTS, NUM_COLS);
        // println(nf(bounds,0,3));

        body_scale = BODY_SCALE_BASE / max(bounds[6], bounds[7], bounds[8]);
        body_trans[0] = -(bounds[3]+bounds[0]) / 2.0;
        body_trans[1] = -(bounds[4]+bounds[1]) / 2.0;
        body_trans[2] = -(bounds[5]+bounds[2]) / 2.0;
        sphere_dia = SPHERE_DIA_BASE; // / max(bounds[6], bounds[7], bounds[8]);
        // println(nf(body_scale,0,3));
        // println(nf(sphere_dia,0,3));
        
    }

    void draw()
    {

        pushStyle_safe();

        ellipseMode(CENTER);
        //colorMode(HSB,100);
        //strokeWeight(2);
        noFill();

        pushMatrix();

        scale(body_scale,-body_scale,-body_scale);
        translate(body_trans[0], body_trans[1], body_trans[2]);

        noStroke();
        fill(255,255,0);

        for (int j = 0; j < NUM_JOINTS; j++) {
            float x1, y1, z1;
            x1 = joints[(step * NUM_COLS) + j * 3 + 0];
            y1 = joints[(step * NUM_COLS) + j * 3 + 1];
            z1 = joints[(step * NUM_COLS) + j * 3 + 2];
            pushMatrix();
            translate(x1, y1, z1);
            sphere(sphere_dia);
            popMatrix();
        }

        stroke(0,0,0);
        strokeWeight(2);

        for (int i = 0; i < links.length / 2; i++) {
            int j_from, j_to;
            j_from = links[i * 2 + 0] - 1; // 1-base to 0-base
            j_to = links[i * 2 + 1] - 1; // 1-base to 0-base
            float x1, y1, z1, x2, y2, z2;
            x1 = joints[(step * NUM_COLS) + j_from * 3 + 0];
            y1 = joints[(step * NUM_COLS) + j_from * 3 + 1];
            z1 = joints[(step * NUM_COLS) + j_from * 3 + 2];
            x2 = joints[(step * NUM_COLS) + j_to * 3 + 0];
            y2 = joints[(step * NUM_COLS) + j_to * 3 + 1];
            z2 = joints[(step * NUM_COLS) + j_to * 3 + 2];
            line(x1, y1, z1, x2, y2, z2);
        }

        popMatrix();

        popStyle_safe();

    }

    public void update(float dt)
    {
        step++;
        if (step >= num_motions) step = 0;
    }

}
