
#include <iostream>
#include <vector>
#include <TH1I.h>
#include <TCanvas.h>
#include <TLine.h>
#include <TROOT.h>
#include <TColor.h>
#include <float.h>

void Fit6_12() {

    float p0, p1, p2, p3, p4, p5;
    float dp1;
    double x;


    fstream fp;
    string line;
    string nome_file = "C:/root_v6.26.14/macros/DT12-12.txt";

   //creazione dello spettro in un range fissato per favorirne la leggibilità
    TH1F* hist = new TH1F("hist", "Vita Media Forti", 200, 0e-6, 25e-6);
    fp.open(nome_file, ios::in);
    if (fp.is_open()) {
        while (getline(fp, line)) {
            x = stod(line);
            hist->Fill(x);
        }
    }
    else {
        std::cout << "non funziona!!!" << std::endl;
    }
    fp.close();
   // TH1F* hist1 = (TH1F*)hist->Clone();
    //hist1->GetXaxis()->SetRangeUser(0.85e-6, 10e-6);

    TCanvas* c1 = new TCanvas();
   // c1->Divide(2);
    //c1->cd(1);

     /*
    TF1* f1 = new TF1("f1", "([0]*exp(-x/[1])) + [2]");
    f1->SetParameters(1000, 2.2e-6,50);
    hist->Fit("f1", "", "", 1.2e-6, 22e-6);
    p0 = f1->GetParameter(0);
    p1 = f1->GetParameter(1);
    p2 = f1->GetParameter(2);
    dp1 = f1->GetParError(1);
    hist->Draw();
    

   
    TF1* f2 = new TF1("f2", "(500*exp(-x/(2e-6))) + ([0]*exp(-x/[1])) + [2]");
    f2->SetParameters(500, 2.5e-6, 30);
    hist->Fit("f2", "", "", 1e-6, 20e-6);
    p0 = f2->GetParameter(0);
    p1 = f2->GetParameter(1);
    p2 = f2->GetParameter(2);
    p3 = f2->GetParameter(3);
   // p4 = f2->GetParameter(4);
    //p5 = f2->GetParameter(5);
   // dp1 = f2->GetParError(1);
    hist->DrawClone();

    //CREAZIONE E STAMPA DELL'ISTOGRAMMA DELLA VITA MEDIA 
    string nome_file2 = "C:/root_v6.26.14/macros/DT6-12new.txt";

    TH1F* hist2 = new TH1F("hist2", "Vita Media", 200, 0e-6, 25e-6);
    fp.open(nome_file2, ios::in);
    if (fp.is_open()) {
        while (getline(fp, line)) {
            x = stod(line);
            hist2->Fill(x);
        }
    }
    else {
        std::cout << "non funziona!!!" << std::endl;
    }
    fp.close();



    //TF1* f1 = new TF1("f1", "([0]*exp(-x/[1])) + [2]");
    f1->SetParameters(1000, 2.2e-6, 100);
    hist2->Fit("f1", "", "", 1.2e-6, 22e-6);
    p0 = f1->GetParameter(0);
    p1 = f1->GetParameter(1);
    p2 = f1->GetParameter(2);
    dp1 = f1->GetParError(1);
   

    TCanvas* c3 = new TCanvas();
    hist2->Draw();
   */ 
  

    TF1* f2 = new TF1("f2", "([0]*exp(-x/[1])) + ([2]*exp(-x/[3])) + [4]");
    f2->SetParameters(1000, 1e-6, 500, 2.5e-6, 30);
    hist->Fit("f2", "", "", 1e-6, 20e-6);
    p0 = f2->GetParameter(0);
    p1 = f2->GetParameter(1);
    p2 = f2->GetParameter(2);
    p3 = f2->GetParameter(3);
    p4 = f2->GetParameter(4);
    //p5 = f2->GetParameter(5);
    dp1 = f2->GetParError(1);
    hist->DrawClone();
    

    //c1->cd(2);
    //hist1 = hist;
    //auto res1 = new TRatioPlot(hist, "diff");
    //TRatioPlot* res1 = new TRatioPlot(hist, "pois");
    //res1->Draw("E");

    //c1->cd(2);
    //TF1* f2 = new TF1("f2", "x-x");
  // hist->Add(f1, -1);
    //hist1->Add(f2, 1);
    //hist->Draw("E");



    gStyle->SetOptFit(0111);
    //creazione del root file che contiene l'istogramma
    TFile out_file("F1.root", "RECREATE");
    //scrittura e chiusura del file
    hist->Write("CleanFORTI");
    //hist2->Write("Normale");
    out_file.Close();

   
}