clear;

load('label.mat');
nc=find(label==0);
scd=find(label==1);

load('att_all.mat')
att=reshape(att,[176,5,5,22]);

%% normalize att
att_z=zeros(size(att));
for s=1:5
    for cv=1:5
        temp=squeeze(att(:,cv,s,:));
        dim=size(temp);
        temp=reshape(temp,[1,dim(1)*dim(2)]);
        temp=zscore(temp);
        temp=reshape(temp,[dim(1),dim(2)]);
        
        % the whole model could utilize an inverse-signed attention (associated with an inverse-signed MLP)
        % which will not influence the seperation of two groups
        % we manually require the attention on SCD to be higher than NC
        if mean(mean(temp(scd,:)))<mean(mean(temp(nc,:)))
            temp=-temp;
        end
        att_z(:,cv,s,:)=reshape(temp,[dim(1),dim(2)]);
    end
end

att_z_m=squeeze(mean(att_z,2));

plot(squeeze(att_z(6,1,:,:)))
hold on;

%% show att dist
s=5;
cv=5;
ncstate=squeeze(att_z(nc,cv,s,:));
ncstate=reshape(ncstate,[1,64*22]);
scdstate=squeeze(att_z(scd,cv,s,:));
scdstate=reshape(scdstate,[1,112*22]);

histogram(ncstate,'normalization','probability')
hold on;
histogram(scdstate,'normalization','probability')

%% mean att across folds

for s=1:5
ncstate=squeeze(att_z_m(nc,s,:));
ncstate=reshape(ncstate,[1,64*22]);
scdstate=squeeze(att_z_m(scd,s,:));
scdstate=reshape(scdstate,[1,112*22]);
[p,h]=ranksum(ncstate,scdstate)
subplot(2,3,s)
histogram(ncstate,'normalization','probability')
hold on;
histogram(scdstate,'normalization','probability')
end
%% plot mean att time series
for sub=1:4
    subplot(2,2,sub)
    plot(squeeze(att_z_m(5+sub,:,:))')
end

%% load FC
% FC_scales=cell(1,5);
% for i=100:100:500
%     load(['att_' num2str(i) '_test.mat'])
%     FC_scales{round(i/100)}=FC_all;
% end

load('FC_scales')

%% group analysis of dFCN
s=1;
FC_all=FC_scales{s};
temp=squeeze(att_z_m(:,s,:));
temp=reshape(temp,[1,176*22]);
FC_all=reshape(FC_all,[176*22,s*100,s*100]);

nc_state=FC_all(temp<=0,:,:);
scd_state=FC_all(temp>0,:,:);

% nc_state=FC_all(nc,:,:,:);
% nc_state=reshape(nc_state,[64*22,s*100,s*100]);
% scd_state=FC_all(scd,:,:,:);
% scd_state=reshape(scd_state,[112*22,s*100,s*100]);

%% mean
figure;
toshow=squeeze(mean(nc_state,1));
for i=1:(s*100)
    toshow(i,i)=0;
end
subplot(2,2,1)
imagesc(toshow,[-0.3,0.7])
title('Averaged NC state')

toshow=squeeze(mean(scd_state,1));
for i=1:(s*100)
    toshow(i,i)=0;
end
subplot(2,2,2)
imagesc(toshow,[-0.3,0.7])
title('Averaged SCD state')

subplot(2,2,3)
imagesc(squeeze(mean(scd_state,1))-squeeze(mean(nc_state,1)))
title('Difference (SCD-NC)')

hh=zeros(s*100,s*100);
pp=zeros(s*100,s*100);
for i=1:(s*100)
    for j=(i+1):(s*100)
        [pp(i,j), hh(i,j)]=ranksum(squeeze(nc_state(:,i,j)),squeeze(scd_state(:,i,j)));
    end
end

th=0.0001/((s*100)*(s*100-1)/2);
newhh=(pp<th);
for i=1:(s*100)
    for j=1:i
        newhh(i,j)=0;
    end
end
subplot(2,2,4)
imagesc(newhh)
title('Significance (upper triangle)')

%% mean sort to RSNs (increase or decrease)
for s=1:5
    FC_all=FC_scales{s};
    temp=squeeze(att_z_m(:,s,:));
    temp=reshape(temp,[1,176*22]);
	% by group
%     FC_all=reshape(FC_all,[176*22,s*100,s*100]);

    nc_state=FC_all(label==0,:,:,:);
    nc_state=reshape(nc_state,[64*22,s*100,s*100]);
    scd_state=FC_all(label==1,:,:,:);
    scd_state=reshape(scd_state,[112*22,s*100,s*100]);
	
	% by group
%     nc_state=FC_all(temp<=0,:,:);
%     scd_state=FC_all(temp>0,:,:);
    
    hh=zeros(s*100,s*100);
    pp=zeros(s*100,s*100);
    for i=1:(s*100)
        for j=(i+1):(s*100)
            [pp(i,j), hh(i,j)]=ranksum(squeeze(nc_state(:,i,j)),squeeze(scd_state(:,i,j)));
        end
    end

    th=0.0001/((s*100)*(s*100-1)/2);
    newhh=(pp<th);
    for i=1:(s*100)
        for j=1:i
            newhh(i,j)=0;
        end
    end
    
    diff=squeeze(mean(scd_state,1))-squeeze(mean(nc_state,1));
    incrhh=newhh&(diff>0);
    decrhh=newhh&(diff<0);

    load('par.mat')
    temp=par(1:(s*100),s);
    temp(((s*100)/2+1):(s*100))=temp(((s*100)/2+1):(s*100))+7;
    okgrid=zeros(14,14);
    RSN1=zeros(14,14);
    for i=1:14
        for j=i:14
            if j==i
                RSN1(i,j)=sum(sum(incrhh(find(temp==i),find(temp==j))))/sum((temp==i))/(sum(temp==j)-1)*2;
            else
                RSN1(i,j)=sum(sum(incrhh(find(temp==i),find(temp==j))))/sum((temp==i))/sum(temp==j);
            end
            okgrid(i,j)=1;
        end
    end

    subplot(2,5,s)
    h=imagesc(RSN1);
    set(h,'alphadata',okgrid);
    colormap('jet');
    
    okgrid=zeros(14,14);
    RSN1=zeros(14,14);
    for i=1:14
        for j=i:14
            if j==i
                RSN1(i,j)=sum(sum(decrhh(find(temp==i),find(temp==j))))/sum((temp==i))/sum(temp==j)*2;
            else
                RSN1(i,j)=sum(sum(decrhh(find(temp==i),find(temp==j))))/sum((temp==i))/sum(temp==j);
            end
              okgrid(i,j)=1;
        end
    end

    subplot(2,5,5+s)
    h=imagesc(RSN1);
    set(h,'alphadata',okgrid);
    colormap('jet');
end
%% std
figure;
toshow=squeeze(std(nc_state,1));
for i=1:(s*100)
    toshow(i,i)=0;
end
subplot(2,2,1)
imagesc(toshow,[0.2,0.40])
title('STD in NC state')

toshow=squeeze(std(scd_state,1));
for i=1:(s*100)
    toshow(i,i)=0;
end
subplot(2,2,2)
imagesc(toshow,[0.2,0.40])
title('STD in SCD state')

subplot(2,2,3)
imagesc(squeeze(std(scd_state,1))-squeeze(std(nc_state,1)))
title('Difference (SCD-NC)')

grouping=zeros(1,3872);
grouping(1:size(nc_state,1))=1; grouping((size(nc_state,1)+1):end)=2;


pp=zeros(s*100,s*100);
for i=1:(s*100)
    for j=(i+1):(s*100)
        pp(i,j)=vartestn([squeeze(nc_state(:,i,j));squeeze(scd_state(:,i,j))],grouping, 'TestType','LeveneQuadratic','Display','off');
    end
end

th=0.0001/((s*100)*(s*100-1)/2);
newhh=(pp<th);
for i=1:(s*100)
    for j=1:i
        newhh(i,j)=0;
    end
end
subplot(2,2,4)
imagesc(newhh)
title('Significance (upper triangle)')

%% std sort to RSNs (increase or decrease)
for s=1:5
    FC_all=FC_scales{s};
    temp=squeeze(att_z_m(:,s,:));
    temp=reshape(temp,[1,176*22]);
	% by group
%     FC_all=reshape(FC_all,[176*22,s*100,s*100]);
    
    nc_state=FC_all(label==0,:,:,:);
    nc_state=reshape(nc_state,[64*22,s*100,s*100]);
    scd_state=FC_all(label==1,:,:,:);
    scd_state=reshape(scd_state,[112*22,s*100,s*100]);

	% by group
%     nc_state=FC_all(temp<=0,:,:);
%     scd_state=FC_all(temp>0,:,:);
    
    pp=zeros(s*100,s*100);
    for i=1:(s*100)
        for j=(i+1):(s*100)
            pp(i,j)=vartestn([squeeze(nc_state(:,i,j));squeeze(scd_state(:,i,j))],grouping, 'TestType','LeveneQuadratic','Display','off');
        end
    end

    th=0.0001/((s*100)*(s*100-1)/2);
    newhh=(pp<th);
    for i=1:(s*100)
        for j=1:i
            newhh(i,j)=0;
        end
    end
    
    diff=squeeze(std(scd_state,1))-squeeze(std(nc_state,1));
    incrhh=newhh&(diff>0);
    decrhh=newhh&(diff<0);

    load('/media/user/4TB/matlab/renji/par.mat')
    temp=par(1:(s*100),s);
    temp((s*100)/2:(s*100))=temp((s*100)/2:(s*100))+7;

    okgrid=zeros(14,14);
    RSN1=zeros(14,14);
    for i=1:14
        for j=i:14
            if j==i
                RSN1(i,j)=sum(sum(incrhh(find(temp==i),find(temp==j))))/sum((temp==i))/sum(temp==j)*2;
            else
                RSN1(i,j)=sum(sum(incrhh(find(temp==i),find(temp==j))))/sum((temp==i))/sum(temp==j);
            end
              okgrid(i,j)=1;
        end
    end

    subplot(2,5,s)
    h=imagesc(RSN1);
    set(h,'alphadata',okgrid);
    colormap('jet');
    
    okgrid=zeros(14,14);
    RSN1=zeros(14,14);
    for i=1:14
        for j=i:14
           if j==i
                RSN1(i,j)=sum(sum(decrhh(find(temp==i),find(temp==j))))/sum((temp==i))/sum(temp==j)*2;
           else
                RSN1(i,j)=sum(sum(decrhh(find(temp==i),find(temp==j))))/sum((temp==i))/sum(temp==j);
           end
              okgrid(i,j)=1;
        end
    end

    subplot(2,5,5+s)
    h=imagesc(RSN1);
    set(h,'alphadata',okgrid);
    colormap('jet');
end

%% network level differences
Q=zeros(5,3872);
Eg=zeros(5,3872);
th=0.2;
for s=1:5
    FC_all=FC_scales{s};
    FC_all=reshape(FC_all,[176*22,s*100,s*100]);
    FC_all(FC_all<=th)=0;
    zerodiag=~eye(s*100,s*100);
    
    for i=1:3872
        G=squeeze(FC_all(i,:,:)).*zerodiag;
        [~,Q(s,i)] = modularity_und(G);
        Eg(s,i) = efficiency_wei(G);
    end
end
save(['networkmetric_th' num2str(th),'.mat'],'Eg','Q');

%modularity
p=[];
for s=1:5
    temp=squeeze(att_z_m(:,s,:));
    temp=reshape(temp,[1,176*22]);
    pos=(temp<=0);
	
    % by group
%     pos=zeros(176,22);
%     pos(label==0,:)=1;
%     pos=reshape(pos,[1,176*22]);
%     pos=(pos==1);
    
    boxplot(Q(s,pos),'Positions',s,'boxstyle','outline','colors','b','Symbol','b+', 'Widths', 0.25)
    hold on;
    scatter(s+0.025*randn(1, length(find(pos))),Q(s,pos),15,'MarkerFaceColor','b','MarkerEdgeColor','b','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)
    boxplot(Q(s,~pos),'Positions',s+0.3,'boxstyle','outline','colors','r', 'Widths', 0.25)
    scatter(s+0.3+0.025*randn(1, length(find(~pos))),Q(s,~pos),15, 'MarkerFaceColor','r','MarkerEdgeColor','r','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)
    
    set(gca,'xlim',[0.5,5.8]);
    set(gca,'ylim',[0,0.57]);
	
	% no thresholding
%     set(gca,'ylim',[0,20]);
    set(gca,'xtick',[(1:5)+0.15])
    set(gca,'xticklabel',{'100 ROIs','200 ROIs','300 ROIs','400 ROIs','500 ROIs'})
    set(gca,'FontSize',20)
    xlabel('Scale')
    ylabel('Modularity')
    p(s)=ranksum(Q(s,pos),Q(s,~pos));
    if ranksum(Q(s,pos),Q(s,~pos))<0.05
       plot([s,s+0.3],[0.50,0.50],'k-','LineWidth',2)
       scatter(s+0.15,0.52,32,'k*')
	   % no thresholding
%        plot([s,s+0.3],[17,17],'k-','LineWidth',2)
%        scatter(s+0.15,18,32,'k*')
    end
    if ranksum(Q(s,pos),Q(s,~pos))<0.01
        scatter(s+0.15,0.54,32,'k*')
		% no thresholding
%       scatter(s+0.15,18.5,32,'k*')
    end
    if ranksum(Q(s,pos),Q(s,~pos))<0.001
        scatter(s+0.15,0.56,32,'k*')
		% no thresholding
%       scatter(s+0.15,19,32,'k*')
    end
end

%global efficiencyc
p=[];
for s=1:5
    temp=squeeze(att_z_m(:,s,:));
    temp=reshape(temp,[1,176*22]);
    pos=(temp<=0);
    
    % by group
%     pos=zeros(176,22);
%     pos(label==0,:)=1;
%     pos=reshape(pos,[1,176*22]);
%     pos=(pos==1);
    
    boxplot(Eg(s,pos),'Positions',s,'boxstyle','outline','colors','b','Symbol','b+', 'Widths', 0.25)
    hold on;
    scatter(s+0.025*randn(1, length(find(pos))),Eg(s,pos),15,'MarkerFaceColor','b','MarkerEdgeColor','b','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)
    boxplot(Eg(s,~pos),'Positions',s+0.3,'boxstyle','outline','colors','r', 'Widths', 0.25)
    scatter(s+0.3+0.025*randn(1, length(find(~pos))),Eg(s,~pos),15, 'MarkerFaceColor','r','MarkerEdgeColor','r','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)
    
    set(gca,'xlim',[0.5,5.8]);
    set(gca,'ylim',[0.2,0.9]);
	% no thresholding
%     set(gca,'ylim',[-0.2,0.1]);
    set(gca,'xtick',[(1:5)+0.15])
    set(gca,'xticklabel',{'100 ROIs','200 ROIs','300 ROIs','400 ROIs','500 ROIs'})
    set(gca,'FontSize',20)
    xlabel('Scale')
    ylabel('Global Efficiency')
    p(s)=ranksum(Eg(s,pos),Eg(s,~pos));
    if ranksum(Eg(s,pos),Eg(s,~pos))<0.05
       plot([s,s+0.3],[0.80,0.80],'k-','LineWidth',2)
       scatter(s+0.15,0.82,32,'k*')
	   % no thresholding
%        plot([s,s+0.3],[0,0],'k-','LineWidth',2)
%        scatter(s+0.15,0.01,32,'k*')
    end
    if ranksum(Eg(s,pos),Eg(s,~pos))<0.01
       scatter(s+0.15,0.84,32,'k*')
	   % no thresholding
%         scatter(s+0.15,0.02,32,'k*')
    end
     if ranksum(Eg(s,pos),Eg(s,~pos))<0.001
         scatter(s+0.15,0.86,32,'k*')
		 % no thresholding
%        scatter(s+0.15,0.03,32,'k*')
    end
end

%% dwell time and frequency group

for s=1:5
    dt=squeeze(mean(att_z_m(:,s,:),3));

    boxplot(dt(nc),'Positions',s,'boxstyle','outline','colors','b','Symbol','b+', 'Widths', 0.25)
    hold on;
    scatter(s+0.025*randn(1, length(dt(nc))),dt(nc),15,'MarkerFaceColor','b','MarkerEdgeColor','b','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)
    boxplot(dt(scd),'Positions',s+0.3,'boxstyle','outline','colors','r', 'Widths', 0.25)
    scatter(s+0.3+0.025*randn(1, length(dt(scd))),dt(scd),15, 'MarkerFaceColor','r','MarkerEdgeColor','r','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)
    
    set(gca,'xtick',[(1:5)+0.15])
    set(gca,'xticklabel',{'100 ROIs','200 ROIs','300 ROIs','400 ROIs','500 ROIs'})
    set(gca,'FontSize',15)
    xlabel('Scale')
    ylabel('Dwell time in SCD-related state')
    p(s)=ranksum(dt(nc),dt(scd));
    if ranksum(dt(nc),dt(scd))<0.05
        plot([s,s+0.3],[1,1],'k-','LineWidth',2)
        scatter(s+0.15,1.1,32,'k*')
    end
    if ranksum(dt(nc),dt(scd))<0.01
        scatter(s+0.15,1.2,32,'k*')
    end
     if ranksum(dt(nc),dt(scd))<0.001
        scatter(s+0.15,1.3,32,'k*')
    end
end
set(gca,'xlim',[0.5,5.8]);
set(gca,'ylim',[-1.5,1.5]);

for s=1:5
    dt=std(squeeze(att_z_m(:,s,:)),0,2);
    boxplot(dt(nc),'Positions',s,'boxstyle','outline','colors','b','Symbol','b+', 'Widths', 0.25)
    hold on;
    scatter(s+0.025*randn(1, length(dt(nc))),dt(nc),15,'MarkerFaceColor','b','MarkerEdgeColor','b','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)
    boxplot(dt(scd),'Positions',s+0.3,'boxstyle','outline','colors','r', 'Widths', 0.25)
    scatter(s+0.3+0.025*randn(1, length(dt(scd))),dt(scd),15, 'MarkerFaceColor','r','MarkerEdgeColor','r','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)
    
    set(gca,'xtick',[(1:5)+0.15])
    set(gca,'xticklabel',{'100 ROIs','200 ROIs','300 ROIs','400 ROIs','500 ROIs'})
    set(gca,'FontSize',13)
    xlabel('Scale')
    ylabel('Transition variability')
    p(s)=ranksum(dt(nc),dt(scd));
    if ranksum(dt(nc),dt(scd))<0.05
       plot([s,s+0.3],[0.51,0.51],'k-','LineWidth',2)
       scatter(s+0.15,0.52,32,'k*')
    end
    if ranksum(dt(nc),dt(scd))<0.01
       scatter(s+0.15,0.54,32,'k*')
    end
     if ranksum(dt(nc),dt(scd))<0.001
       scatter(s+0.15,0.56,32,'k*')
    end
end
set(gca,'xlim',[0.5,5.8]);
set(gca,'ylim',[0,0.6]);

%% dwell time and frequency group cognition
s=3;
% dt=squeeze(mean(att_z_m(:,s,:),3));
dt=std(squeeze(att_z_m(:,s,:)),0,2);

load('metrics_sort.mat')

r=zeros(1,9);
p=zeros(1,9);
CI=zeros(2,9);
for i=1:9
    x=dt(:); y=metrics(:,i);
    [r(i),p(i)]=corr(x,y,'rows','complete','Type','Spearman');
    mycorr=@(x,y) (corr(x,y,'rows','complete','Type','Spearman'));
    CI(:,i)=bootci(1000,{mycorr,x,y});
end

testname={'MMSE','MOCA','AVLTN5','AVLTN7','AFT','BNT','STTA','STTB','FAQ'};
[~,pcor,padj] = fdr(p);

% scatter to check
for i=1:9
    if pcor(i)<0.05
        figure;
        scatter(dt,metrics(:,i)+0.3*rand(length(dt),1),'filled');
        alpha(0.5);
        hold on;
        c=ones(length(dt),1);
        b = regress(metrics(:,i),[dt,c]);
        yf= [dt,c]*b;
        plot(dt,yf)
        xlabel(['attention mean (scale)=' num2str(s*100)]);
%         xlabel(['attention STD (scale)=' num2str(s*100)]);
        ylabel(testname{i});
        title(['r=' num2str(r(i),3) ' p=' num2str(pcor(i),3)]);
    end
end