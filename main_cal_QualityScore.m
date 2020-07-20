function main_cal_QualityScore

type_test = "JPEG"; % HEVC or JPEG

dir_img = "out/img_" + type_test;
fp = fopen("out/report_QualityScore_" + type_test + ".txt", "w");
if type_test == "HEVC"
    list_QPorQF = ["22","27","32","37","42"];
    tab = "QP";
elseif type_test == "JPEG"
    list_QPorQF = ["50","40","30","20","10"];
    tab = "QF";
end
num_test = 5; %1000;

for ite_img = 1:num_test
    
    for QPorQF = list_QPorQF
        
        for output = ["cmp", "1", "2", "3", "4", "5"]
            
            if output == "cmp"
                type = tab + QPorQF + "_cmp";
            else
                type = tab + QPorQF + "_out" + output;
            end
            
            img_path = fullfile(dir_img, string(ite_img-1) + "_" + type + ".bmp");
            img = imread(img_path);
            
            score = cal_QualityScore(img, type_test);
            
            fprintf("%d - %s %s - output %s - %.2f\n", ite_img, tab, QPorQF, output, score);
            fprintf(fp, "%d - %s %s - output %s - %.2f\n", ite_img, tab, QPorQF, output, score);
            
        end
        
        fprintf("\n");
        
    end
end

fclose(fp);

end