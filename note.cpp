// Function that returns the coordinates of boxes in the scene
std::vector<Vec3d> computeAvgBoxes(std::vector<Vec3d> rMaster, std::vector<Vec3d> tMaster, std::vector<bool> init_id) {
    std::vector<Vec3d> avg_points;
    Vec3d a0, b0, c0, d0, a1, b1, c1, d1, a3, b3, c3, d3;
    std::vector<Vec3d> a_sum, b_sum, c_sum, d_sum;

    a0[0] = -1.6;
    a0[1] = -10.7 + 0.5;
    a0[2] = -3;
    b0[0] = -1.6;
    b0[1] = -10.7 + 0.5;
    b0[2] = -43;
    c0[0] = -1.6;
    c0[1] = 9.3 + 0.5;
    c0[2] = -23;
    d0[0] = -1.6;
    d0[1] = -30.7 + 0.5;
    d0[2] = -23;    
    a1[0] = 0.0 - 0.5;
    a1[1] = -1.5;
    a1[2] = -3;
    b1[0] = 0.0 - 0.5;
    b1[1] = -1.5;
    b1[2] = -43;
    c1[0] = -20 - 0.5;
    c1[1] = -1.5;
    c1[2] = -23;
    d1[0] = 20 - 0.5;
    d1[1] = -1.5;
    d1[2] = -23;
            
    a0 = transformVec(a0, rMaster[0], tMaster[0]);
    b0 = transformVec(b0, rMaster[0], tMaster[0]);
    c0 = transformVec(c0, rMaster[0], tMaster[0]);
    d0 = transformVec(d0, rMaster[0], tMaster[0]);    
    a1 = transformVec(a1, rMaster[1], tMaster[1]);
    b1 = transformVec(b1, rMaster[1], tMaster[1]);
    c1 = transformVec(c1, rMaster[1], tMaster[1]);
    d1 = transformVec(d1, rMaster[1], tMaster[1]);

    if(init_id[0]) {
        a_sum.push_back(a0);
        b_sum.push_back(b0);
        c_sum.push_back(c0);
        d_sum.push_back(d0);
    }
    if(init_id[4]) {
        a_sum.push_back(a1);
        b_sum.push_back(b1);
        c_sum.push_back(c1);
        d_sum.push_back(d1);
    }
    if(init_id[0]||init_id[4]){
        for (int i=0; i<3; i++) {
            a_avg[i] = 0.0;
            b_avg[i] = 0.0;
            c_avg[i] = 0.0;
            d_avg[i] = 0.0;
            for (unsigned int j=0; j<a_sum.size(); j++) {
                a_avg[i] += a_sum[j][i];
                b_avg[i] += b_sum[j][i];
                c_avg[i] += c_sum[j][i];
                d_avg[i] += d_sum[j][i];
            }
            a_avg[i] /= a_sum.size();
            b_avg[i] /= b_sum.size();
            c_avg[i] /= c_sum.size();
            d_avg[i] /= d_sum.size();
        }
    }
    else {
        return avg_points;
    }
    avg_points.push_back(a_avg);
    avg_points.push_back(b_avg);
    avg_points.push_back(c_avg);
    avg_points.push_back(d_avg);
    
    return avg_points;
} 

