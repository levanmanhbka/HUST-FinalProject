# Generated by Django 2.2.1 on 2019-05-18 07:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_lanmark_num'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lanmark',
            name='image',
            field=models.ImageField(null=True, upload_to=''),
        ),
    ]