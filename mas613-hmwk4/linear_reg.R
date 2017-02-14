
data.train = read.csv("part1 train.csv",header = FALSE)
data.test = read.csv("part1 test.csv",header = FALSE)
colnames(data.train)<- c("x","y")
colnames(data.test)<- c("x","y")
summary(data.train)

plot(data.train$x,data.train$y)

fit = lm(y~x, data = data.train)

summary(fit)

pred = predict(fit, newdata = data.frame(x =data.test$x))
error = pred - data.test$y

mse = mean(error^2)

data = read.csv("part2.csv",header = F)
fdata = scale(data)
fdata = data.frame(fdata)
class(fdata)
colnames(fdata) <- c("age","mother_edu","father_edu","travel_time","study_time","past_failure",
                     "family_rel","free_time","going_out","weekday_alcohol","weekend_alcohol",
                     "health_status","school_abs","grade")

plot(fdata$age,fdata$grade)
plot(fdata$mother_edu,fdata$grade)
plot(fdata$father_edu,fdata$grade)
plot(fdata$travel_time,fdata$grade)
plot(fdata$study_time,fdata$grade)
plot(fdata$past_failure,fdata$grade)
plot(fdata$family_rel,fdata$grade)
plot(fdata$free_time,fdata$grade)
plot(fdata$going_out,fdata$grade)
plot(fdata$weekday_alcohol,fdata$grade)
plot(fdata$weekend_alcohol,fdata$grade)
plot(fdata$health_status,fdata$grade)
plot(fdata$school_abs,fdata$grade)

fit = lm(grade~.,data = fdata)
plot(fit)
summary(fit)
cor(fdata,y=fdata$grade)
